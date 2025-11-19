"""
GPU-Accelerated TF-Denoise Processor.

Extends the CPU-based TF-Denoise processor with GPU acceleration using PyTorch
for both MPS (MacBook) and CUDA (NVIDIA) backends, with automatic CPU fallback.
"""

import numpy as np
import torch
import warnings
from typing import Optional, Literal
import logging
import time

import sys

from models.seismic_data import SeismicData
from processors.tf_denoise import TFDenoise
from processors.gpu.device_manager import DeviceManager, get_device_manager
from processors.gpu.stft_gpu import STFT_GPU
from processors.gpu.stransform_gpu import STransformGPU
from processors.gpu.thresholding_gpu import ThresholdingGPU
from processors.gpu.utils_gpu import numpy_to_tensor, tensor_to_numpy

logger = logging.getLogger(__name__)


class TFDenoiseGPU(TFDenoise):
    """
    GPU-accelerated TF-domain denoising processor.

    Extends TFDenoise with GPU acceleration while maintaining full API compatibility.
    Automatically falls back to CPU processing if GPU is unavailable or encounters errors.
    """

    def __init__(
        self,
        aperture: int = 7,
        fmin: float = 5.0,
        fmax: float = 100.0,
        threshold_k: float = 3.0,
        threshold_type: Literal['soft', 'garrote'] = 'soft',
        transform_type: Literal['stransform', 'stft'] = 'stransform',
        use_gpu: Literal['auto', 'force', 'never'] = 'auto',
        device_manager: Optional[DeviceManager] = None
    ):
        """
        Initialize GPU-accelerated TF-Denoise processor.

        Args:
            aperture: Spatial aperture size (number of traces, must be odd)
            fmin: Minimum frequency (Hz) for processing
            fmax: Maximum frequency (Hz) for processing
            threshold_k: MAD threshold multiplier
            threshold_type: Type of thresholding ('soft' or 'garrote')
            transform_type: Transform to use ('stransform' or 'stft')
            use_gpu: GPU usage mode:
                - 'auto': Use GPU if available, fall back to CPU (default)
                - 'force': Use GPU or raise error
                - 'never': Always use CPU
            device_manager: Optional DeviceManager instance (created if None)
        """
        # Initialize parent class (CPU version)
        super().__init__(
            aperture=aperture,
            fmin=fmin,
            fmax=fmax,
            threshold_k=threshold_k,
            threshold_type=threshold_type,
            transform_type=transform_type
        )

        # GPU configuration
        self.use_gpu_mode = use_gpu

        # Initialize device manager
        if device_manager is None:
            enable_gpu = (use_gpu != 'never')
            self.device_manager = get_device_manager(enable_gpu=enable_gpu)
        else:
            self.device_manager = device_manager

        self.device = self.device_manager.device

        # Initialize GPU processors
        if self.device_manager.is_gpu_available() and use_gpu != 'never':
            self._init_gpu_processors()
            logger.info(f"TF-Denoise GPU initialized on {self.device_manager.get_device_name()}")
        else:
            self.gpu_stft = None
            self.gpu_stransform = None
            self.gpu_thresholding = None
            logger.info("TF-Denoise running in CPU-only mode")

    def _init_gpu_processors(self):
        """Initialize GPU processing modules."""
        self.gpu_stft = STFT_GPU(device=self.device)
        self.gpu_stransform = STransformGPU(device=self.device)
        self.gpu_thresholding = ThresholdingGPU(
            device=self.device,
            threshold_k=self.threshold_k,
            threshold_type=self.threshold_type
        )

    def get_description(self) -> str:
        """Get processor description with GPU status."""
        base_desc = super().get_description()
        if self.device_manager.is_gpu_available() and self.use_gpu_mode != 'never':
            gpu_name = self.device_manager.get_device_name()
            return f"{base_desc} [GPU: {gpu_name}]"
        else:
            return f"{base_desc} [CPU]"

    def process(self, data: SeismicData) -> SeismicData:
        """
        Apply GPU-accelerated TF-domain denoising.

        Automatically falls back to CPU if GPU processing fails.

        Args:
            data: Input seismic data

        Returns:
            Denoised seismic data
        """
        # If GPU disabled or unavailable, use CPU
        if self.use_gpu_mode == 'never' or not self.device_manager.is_gpu_available():
            return super().process(data)

        # Try GPU processing
        try:
            return self._process_gpu(data)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if self.use_gpu_mode == 'force':
                # Re-raise error if GPU forced
                raise RuntimeError(f"GPU processing failed (force mode): {e}") from e
            else:
                # Fall back to CPU
                warnings.warn(
                    f"GPU processing failed, falling back to CPU: {e}",
                    RuntimeWarning
                )
                logger.warning(f"GPU processing failed, using CPU fallback: {e}")
                return super().process(data)

    def _process_gpu(self, data: SeismicData) -> SeismicData:
        """
        GPU-accelerated processing implementation.

        Args:
            data: Input seismic data

        Returns:
            Denoised seismic data
        """
        print(f"\n{'='*60}")
        print(f"TF-DENOISE GPU - Starting processing")
        print(f"{'='*60}")

        start_time_total = time.time()

        traces = data.traces.copy()
        n_samples, n_traces = traces.shape

        print(f"Device: {self.device_manager.get_device_name()}")
        print(f"Input data: {n_samples} samples × {n_traces} traces")
        print(f"Parameters:")
        print(f"  - Aperture: {self.aperture}")
        print(f"  - Frequency range: {self.fmin:.1f}-{self.fmax:.1f} Hz")
        print(f"  - Threshold k: {self.threshold_k}")
        print(f"  - Threshold type: {self.threshold_type}")
        print(f"  - Transform: {self.transform_type}")

        # Check if data fits in GPU memory
        can_fit = self.device_manager.can_fit_data(n_samples, n_traces)
        if not can_fit:
            print(f"⚠️  Warning: Dataset may exceed GPU memory, using batching")

        # Calculate batch size
        batch_size = self.device_manager.calculate_batch_size(n_samples, n_traces)
        print(f"Batch size: {batch_size} traces")

        # Validate aperture
        if n_traces < self.aperture:
            print(f"⚠️  Warning: Not enough traces ({n_traces}) for aperture ({self.aperture}). "
                  f"Using all available traces.")
            effective_aperture = n_traces if n_traces % 2 == 1 else n_traces - 1
        else:
            effective_aperture = self.aperture

        # Process traces with spatial aperture
        denoised_traces = np.zeros_like(traces)

        # Get sampling frequency in Hz
        # data.sample_rate is the sample interval in milliseconds
        # Sampling frequency (Hz) = 2 × Nyquist frequency
        nyquist_freq = data.nyquist_freq
        sample_rate = 2.0 * nyquist_freq  # Convert to sampling frequency in Hz

        print(f"Sample interval: {data.sample_rate:.1f} ms")
        print(f"Sampling frequency: {sample_rate:.1f} Hz")
        print(f"Nyquist frequency: {nyquist_freq:.1f} Hz")

        # Process based on transform type
        if self.transform_type == 'stransform':
            denoised_traces = self._process_with_stransform_gpu(
                traces,
                effective_aperture,
                sample_rate
            )
        else:  # stft
            denoised_traces = self._process_with_stft_gpu(
                traces,
                effective_aperture,
                sample_rate
            )

        # Report timing
        elapsed_total = time.time() - start_time_total
        print(f"\n{'='*60}")
        print(f"✓ GPU Processing completed in {elapsed_total:.3f} seconds")
        print(f"  Throughput: {n_traces / elapsed_total:.1f} traces/sec")
        print(f"{'='*60}\n")

        # Energy verification
        input_rms = np.sqrt(np.mean(traces**2))
        output_rms = np.sqrt(np.mean(denoised_traces**2))
        ratio = output_rms / input_rms if input_rms > 0 else 0

        print(f"ENERGY VERIFICATION:")
        print(f"  Input RMS:  {input_rms:.6f}")
        print(f"  Output RMS: {output_rms:.6f}")
        print(f"  Ratio:      {ratio:.2%}")

        if ratio < 0.10:
            print(f"  ⚠️  WARNING: Output is < 10% of input - threshold may be too aggressive!")
        elif 0.70 <= ratio <= 0.95:
            print(f"  ✓ Output is signal model (70-95% of input energy preserved)")
        elif ratio > 0.95:
            print(f"  ⚠️  WARNING: Output is > 95% of input - minimal denoising occurred")

        print(f"{'='*60}\n")

        # Clear GPU cache
        self.device_manager.clear_cache()

        # Create output SeismicData
        return SeismicData(
            traces=denoised_traces,
            sample_rate=data.sample_rate,
            metadata={
                **data.metadata,
                'processed': True,
                'processor': 'TF-Denoise GPU',
                'gpu_device': self.device_manager.get_device_name(),
                'processing_time': elapsed_total
            }
        )

    def _process_with_stransform_gpu(
        self,
        traces: np.ndarray,
        aperture: int,
        sample_rate: float
    ) -> np.ndarray:
        """
        Process using GPU-accelerated S-Transform.

        Args:
            traces: Input traces (n_samples, n_traces)
            aperture: Spatial aperture size
            sample_rate: Sample rate in Hz

        Returns:
            Denoised traces
        """
        n_samples, n_traces = traces.shape
        denoised_traces = np.zeros_like(traces)
        half_aperture = aperture // 2

        print(f"\n📊 S-Transform GPU processing:")
        print(f"   Frequency range: {self.fmin:.1f}-{self.fmax:.1f} Hz")

        start_time = time.time()

        # Process each trace with its spatial aperture
        for i in range(n_traces):
            # Get aperture indices
            start_idx = max(0, i - half_aperture)
            end_idx = min(n_traces, i + half_aperture + 1)
            ensemble = traces[:, start_idx:end_idx]

            # Compute S-Transforms for ensemble on GPU
            st_ensemble = []
            for j in range(ensemble.shape[1]):
                S, freqs = self.gpu_stransform.forward(
                    ensemble[:, j],
                    fmin=self.fmin,
                    fmax=self.fmax,
                    sample_rate=sample_rate
                )
                st_ensemble.append(S)

            st_ensemble = np.array(st_ensemble)  # (n_traces_aperture, n_freqs, n_times)

            # Center trace S-Transform
            center_idx = i - start_idx
            st_center = st_ensemble[center_idx]

            # Apply MAD thresholding on GPU
            st_denoised = self.gpu_thresholding.apply_mad_thresholding(
                st_center,
                st_ensemble,
                spatial_dim=0
            )

            # Inverse S-Transform
            denoised_traces[:, i] = self.gpu_stransform.inverse(
                st_denoised,
                freqs,
                n_samples,
                sample_rate
            )

            # Progress reporting
            if (i + 1) % 50 == 0 or (i + 1) == n_traces:
                elapsed = time.time() - start_time
                progress = (i + 1) / n_traces * 100
                traces_per_sec = (i + 1) / elapsed
                print(f"   Progress: {i+1}/{n_traces} ({progress:.1f}%) | "
                      f"{traces_per_sec:.1f} traces/sec")

        elapsed_total = time.time() - start_time
        print(f"   ✓ S-Transform GPU completed: {elapsed_total:.3f}s "
              f"({elapsed_total/n_traces*1000:.1f}ms per trace)")

        return denoised_traces

    def _process_with_stft_gpu(
        self,
        traces: np.ndarray,
        aperture: int,
        sample_rate: float
    ) -> np.ndarray:
        """
        Process using GPU-accelerated STFT.

        Args:
            traces: Input traces (n_samples, n_traces)
            aperture: Spatial aperture size
            sample_rate: Sample rate in Hz

        Returns:
            Denoised traces
        """
        n_samples, n_traces = traces.shape
        denoised_traces = np.zeros_like(traces)
        half_aperture = aperture // 2

        print(f"\n📊 STFT GPU processing:")
        print(f"   Frequency range: {self.fmin:.1f}-{self.fmax:.1f} Hz")

        start_time = time.time()

        # Process each trace with its spatial aperture
        for i in range(n_traces):
            # Get aperture indices
            start_idx = max(0, i - half_aperture)
            end_idx = min(n_traces, i + half_aperture + 1)
            ensemble = traces[:, start_idx:end_idx]

            # Compute STFTs for ensemble on GPU
            stft_ensemble = []
            for j in range(ensemble.shape[1]):
                S, freqs = self.gpu_stft.forward(
                    ensemble[:, j],
                    fmin=self.fmin,
                    fmax=self.fmax,
                    sample_rate=sample_rate
                )
                stft_ensemble.append(S)

            stft_ensemble = np.array(stft_ensemble)  # (n_traces_aperture, n_freqs, n_frames)

            # Center trace STFT
            center_idx = i - start_idx
            stft_center = stft_ensemble[center_idx]

            # Apply MAD thresholding on GPU
            stft_denoised = self.gpu_thresholding.apply_mad_thresholding(
                stft_center,
                stft_ensemble,
                spatial_dim=0
            )

            # Inverse STFT
            denoised_traces[:, i] = self.gpu_stft.inverse(
                stft_denoised,
                signal_length=n_samples
            )

            # Progress reporting
            if (i + 1) % 50 == 0 or (i + 1) == n_traces:
                elapsed = time.time() - start_time
                progress = (i + 1) / n_traces * 100
                traces_per_sec = (i + 1) / elapsed
                print(f"   Progress: {i+1}/{n_traces} ({progress:.1f}%) | "
                      f"{traces_per_sec:.1f} traces/sec")

        elapsed_total = time.time() - start_time
        print(f"   ✓ STFT GPU completed: {elapsed_total:.3f}s "
              f"({elapsed_total/n_traces*1000:.1f}ms per trace)")

        return denoised_traces
