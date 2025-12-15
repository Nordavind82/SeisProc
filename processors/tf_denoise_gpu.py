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

    Supports multiple threshold modes for improved noise removal:
    - 'soft': Classical soft thresholding (partial removal, legacy)
    - 'hard': Full removal for outliers, preserve non-outliers exactly
    - 'scaled': Progressive removal based on outlier severity
    - 'adaptive': Combined hard (severe) + scaled (moderate) - recommended
    """

    def __init__(
        self,
        aperture: int = 7,
        fmin: float = 5.0,
        fmax: float = 100.0,
        threshold_k: float = 3.0,
        threshold_type: Literal['soft', 'garrote'] = 'soft',
        threshold_mode: Literal['soft', 'hard', 'scaled', 'adaptive'] = 'adaptive',
        transform_type: Literal['stransform', 'stft'] = 'stransform',
        use_gpu: Literal['auto', 'force', 'never'] = 'auto',
        low_amp_protection: bool = True,
        low_amp_factor: float = 0.3,
        time_smoothing: int = 1,
        device_manager: Optional[DeviceManager] = None
    ):
        """
        Initialize GPU-accelerated TF-Denoise processor.

        Args:
            aperture: Spatial aperture size (number of traces, must be odd)
            fmin: Minimum frequency (Hz) for processing
            fmax: Maximum frequency (Hz) for processing
            threshold_k: MAD threshold multiplier
            threshold_type: Type of thresholding ('soft' or 'garrote') - legacy parameter
            threshold_mode: Noise removal mode (recommended: 'adaptive'):
                - 'soft': Classical soft thresholding (partial removal)
                - 'hard': Full removal for outliers (Option A)
                - 'scaled': Progressive removal based on severity (Option B)
                - 'adaptive': Hard for severe + scaled for moderate (recommended)
            transform_type: Transform to use ('stransform' or 'stft')
            use_gpu: GPU usage mode:
                - 'auto': Use GPU if available, fall back to CPU (default)
                - 'force': Use GPU or raise error
                - 'never': Always use CPU
            low_amp_protection: Prevent inflation of low-amplitude samples
            low_amp_factor: Threshold for low-amplitude protection (fraction of median)
            time_smoothing: Time window size for MAD smoothing (1=no smoothing)
            device_manager: Optional DeviceManager instance (created if None)
        """
        # Initialize parent class with all serializable params
        super().__init__(
            aperture=aperture,
            fmin=fmin,
            fmax=fmax,
            threshold_k=threshold_k,
            threshold_type=threshold_type,
            threshold_mode=threshold_mode,
            transform_type=transform_type,
            low_amp_protection=low_amp_protection,
            low_amp_factor=low_amp_factor,
            time_smoothing=time_smoothing
        )

        # GPU configuration (stored separately, added to params for serialization)
        self.use_gpu_mode = use_gpu
        self.params['use_gpu'] = use_gpu  # Add to params for serialization
        self._oom_retry_count = 0
        self._max_oom_retries = 3

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
            threshold_type=self.threshold_type,
            threshold_mode=self.threshold_mode,
            low_amp_protection=self.low_amp_protection,
            low_amp_factor=self.low_amp_factor
        )

    def get_description(self) -> str:
        """Get processor description with GPU status."""
        base_desc = super().get_description()
        mode_str = f", mode={self.threshold_mode}"
        if self.low_amp_protection:
            mode_str += ", low_amp_protect"
        if self.device_manager.is_gpu_available() and self.use_gpu_mode != 'never':
            gpu_name = self.device_manager.get_device_name()
            return f"{base_desc}{mode_str} [GPU: {gpu_name}]"
        else:
            return f"{base_desc}{mode_str} [CPU]"

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

        # Try GPU processing with OOM recovery
        try:
            return self._process_gpu(data)
        except torch.cuda.OutOfMemoryError as e:
            # Specific handling for CUDA out-of-memory
            return self._handle_oom_error(data, e)
        except RuntimeError as e:
            # Check if this is an OOM error wrapped in RuntimeError
            error_msg = str(e).lower()
            if 'out of memory' in error_msg or 'cuda' in error_msg:
                return self._handle_oom_error(data, e)

            if self.use_gpu_mode == 'force':
                raise RuntimeError(f"GPU processing failed (force mode): {e}") from e
            else:
                warnings.warn(
                    f"GPU processing failed, falling back to CPU: {e}",
                    RuntimeWarning
                )
                logger.warning(f"GPU processing failed, using CPU fallback: {e}")
                return super().process(data)
        except Exception as e:
            if self.use_gpu_mode == 'force':
                raise RuntimeError(f"GPU processing failed (force mode): {e}") from e
            else:
                logger.warning(f"Unexpected GPU error, using CPU fallback: {e}", exc_info=True)
                return super().process(data)

    def _handle_oom_error(self, data: SeismicData, error: Exception) -> SeismicData:
        """
        Handle GPU out-of-memory errors with retry logic.

        Attempts to recover by:
        1. Clearing GPU cache
        2. Reducing effective batch size (by processing smaller spatial apertures)
        3. Falling back to CPU after max retries

        Args:
            data: Input seismic data
            error: The OOM exception that occurred

        Returns:
            Processed seismic data (from retry or CPU fallback)
        """
        self._oom_retry_count += 1
        logger.warning(
            f"GPU OOM error (attempt {self._oom_retry_count}/{self._max_oom_retries}): {error}"
        )

        # Clear GPU cache to free memory
        self.device_manager.clear_cache()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        if self._oom_retry_count < self._max_oom_retries:
            # Try reducing aperture to lower memory usage
            original_aperture = self.aperture
            self.aperture = max(3, self.aperture - 2)  # Reduce aperture by 2, minimum 3

            logger.info(
                f"Retrying GPU processing with reduced aperture: {original_aperture} → {self.aperture}"
            )
            print(f"⚠️  OOM recovery: reducing aperture from {original_aperture} to {self.aperture}")

            try:
                result = self._process_gpu(data)
                # Restore original aperture after successful processing
                self.aperture = original_aperture
                self._oom_retry_count = 0
                return result
            except (torch.cuda.OutOfMemoryError, RuntimeError) as retry_error:
                # Restore aperture and try again or fall back
                self.aperture = original_aperture
                return self._handle_oom_error(data, retry_error)
        else:
            # Max retries exceeded - fall back to CPU
            self._oom_retry_count = 0

            if self.use_gpu_mode == 'force':
                raise RuntimeError(
                    f"GPU OOM after {self._max_oom_retries} retries. "
                    f"Data too large for GPU memory."
                ) from error

            logger.warning(
                f"GPU OOM after {self._max_oom_retries} retries, falling back to CPU"
            )
            print(f"⚠️  GPU memory exhausted after retries, using CPU fallback")
            return super().process(data)

    def _process_gpu(self, data: SeismicData) -> SeismicData:
        """
        GPU-accelerated processing implementation.

        Args:
            data: Input seismic data

        Returns:
            Denoised seismic data
        """
        start_time_total = time.time()

        traces = data.traces.copy()
        n_samples, n_traces = traces.shape

        # Log gather summary once at start
        logger.info(
            f"TFD-GPU: {n_traces} traces × {n_samples} samples | "
            f"Device: {self.device_manager.get_device_name()} | "
            f"Transform: {self.transform_type} | "
            f"Aperture: {self.aperture} | "
            f"Freq: {self.fmin:.0f}-{self.fmax:.0f}Hz | "
            f"k={self.threshold_k}, mode={self.threshold_mode}"
        )

        # Check if data fits in GPU memory
        can_fit = self.device_manager.can_fit_data(n_samples, n_traces)
        if not can_fit:
            logger.warning("Dataset may exceed GPU memory, using batching")

        # Calculate batch size
        batch_size = self.device_manager.calculate_batch_size(n_samples, n_traces)

        # Validate aperture
        if n_traces < self.aperture:
            logger.warning(
                f"Not enough traces ({n_traces}) for aperture ({self.aperture}), "
                f"using {n_traces if n_traces % 2 == 1 else n_traces - 1}"
            )
            effective_aperture = n_traces if n_traces % 2 == 1 else n_traces - 1
        else:
            effective_aperture = self.aperture

        # Process traces with spatial aperture
        denoised_traces = np.zeros_like(traces)

        # Get sampling frequency in Hz
        nyquist_freq = data.nyquist_freq
        sample_rate = 2.0 * nyquist_freq

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

        # Compute timing and energy metrics
        elapsed_total = time.time() - start_time_total
        throughput = n_traces / elapsed_total if elapsed_total > 0 else 0

        # Energy verification
        input_rms = np.sqrt(np.mean(traces**2))
        output_rms = np.sqrt(np.mean(denoised_traces**2))
        energy_ratio = output_rms / input_rms if input_rms > 0 else 0

        # Log completion summary with bottleneck metrics
        logger.info(
            f"TFD-GPU complete: {elapsed_total:.2f}s | "
            f"{throughput:.1f} traces/s | "
            f"{elapsed_total/n_traces*1000:.1f}ms/trace | "
            f"Energy: {energy_ratio:.1%} retained"
        )

        # Warn about potential issues
        if energy_ratio < 0.10:
            logger.warning(f"Output <10% of input energy - threshold may be too aggressive")
        elif energy_ratio > 0.95:
            logger.warning(f"Output >95% of input energy - minimal denoising occurred")

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
        Process using GPU-accelerated S-Transform with batch processing.

        Uses batched GPU operations for maximum throughput:
        - Batch FFT for all traces
        - Batch thresholding across spatial aperture
        - Batch inverse transform

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

        start_time = time.time()

        # Timing accumulators for bottleneck analysis
        time_forward = 0.0
        time_threshold = 0.0
        time_inverse = 0.0

        # Determine optimal batch size based on available GPU memory
        batch_size = self.device_manager.calculate_batch_size(n_samples, n_traces)
        batch_size = min(batch_size, 100)  # Cap at 100 traces per batch

        # Process in batches of center traces
        for batch_start in range(0, n_traces, batch_size):
            batch_end = min(batch_start + batch_size, n_traces)
            batch_indices = range(batch_start, batch_end)

            for i in batch_indices:
                # Get aperture indices
                start_idx = max(0, i - half_aperture)
                end_idx = min(n_traces, i + half_aperture + 1)
                ensemble = traces[:, start_idx:end_idx]
                center_idx = i - start_idx

                # Forward S-Transform
                t0 = time.time()
                st_ensemble, freqs = self.gpu_stransform.batch_forward(
                    ensemble,
                    fmin=self.fmin,
                    fmax=self.fmax,
                    sample_rate=sample_rate
                )
                time_forward += time.time() - t0

                # Apply batch MAD thresholding
                t0 = time.time()
                st_denoised_batch = self.gpu_thresholding.apply_batch_mad_thresholding(
                    st_ensemble,
                    spatial_dim=0
                )
                time_threshold += time.time() - t0

                # Extract center trace and inverse transform
                t0 = time.time()
                st_denoised = st_denoised_batch[center_idx]
                denoised_traces[:, i] = self.gpu_stransform.inverse(
                    st_denoised,
                    freqs,
                    n_samples,
                    sample_rate
                )
                time_inverse += time.time() - t0

            # Report to callback for UI updates (no print spam)
            progress = batch_end / n_traces * 100
            self._report_progress(batch_end, n_traces, f"S-Transform denoising: {progress:.0f}%")

        elapsed_total = time.time() - start_time

        # Log timing breakdown for bottleneck identification
        logger.debug(
            f"S-Transform timing: Forward={time_forward:.2f}s ({time_forward/elapsed_total*100:.0f}%) | "
            f"Threshold={time_threshold:.2f}s ({time_threshold/elapsed_total*100:.0f}%) | "
            f"Inverse={time_inverse:.2f}s ({time_inverse/elapsed_total*100:.0f}%)"
        )

        return denoised_traces

    def _process_with_stransform_gpu_sliding_window(
        self,
        traces: np.ndarray,
        aperture: int,
        sample_rate: float
    ) -> np.ndarray:
        """
        Process using GPU-accelerated S-Transform with sliding window batch.

        Alternative batch approach: compute S-Transform for ALL traces once,
        then apply sliding window thresholding.

        Args:
            traces: Input traces (n_samples, n_traces)
            aperture: Spatial aperture size
            sample_rate: Sample rate in Hz

        Returns:
            Denoised traces
        """
        n_samples, n_traces = traces.shape
        half_aperture = aperture // 2

        start_time = time.time()

        # Compute S-Transform for ALL traces at once
        st_all, freqs = self.gpu_stransform.batch_forward(
            traces,
            fmin=self.fmin,
            fmax=self.fmax,
            sample_rate=sample_rate
        )
        time_forward = time.time() - start_time

        # Apply sliding window thresholding
        t0 = time.time()
        denoised_st = np.zeros_like(st_all)

        for i in range(n_traces):
            start_idx = max(0, i - half_aperture)
            end_idx = min(n_traces, i + half_aperture + 1)
            center_idx = i - start_idx

            st_ensemble = st_all[start_idx:end_idx]
            st_denoised_batch = self.gpu_thresholding.apply_batch_mad_thresholding(
                st_ensemble,
                spatial_dim=0
            )
            denoised_st[i] = st_denoised_batch[center_idx]

        time_threshold = time.time() - t0

        # Batch inverse S-Transform
        t0 = time.time()
        denoised_traces = self.gpu_stransform.batch_inverse(
            denoised_st,
            freqs,
            n_samples,
            sample_rate
        )
        time_inverse = time.time() - t0

        elapsed_total = time.time() - start_time

        # Log timing breakdown for bottleneck identification
        logger.debug(
            f"S-Transform sliding: Forward={time_forward:.2f}s ({time_forward/elapsed_total*100:.0f}%) | "
            f"Threshold={time_threshold:.2f}s ({time_threshold/elapsed_total*100:.0f}%) | "
            f"Inverse={time_inverse:.2f}s ({time_inverse/elapsed_total*100:.0f}%)"
        )

        return denoised_traces

    def _process_with_stft_gpu(
        self,
        traces: np.ndarray,
        aperture: int,
        sample_rate: float
    ) -> np.ndarray:
        """
        Process using GPU-accelerated STFT with BULK processing.

        Optimized approach:
        1. Compute STFT for ALL traces at once (single GPU transfer)
        2. Apply sliding window MAD thresholding on GPU
        3. Batch inverse STFT (single GPU transfer back)

        This minimizes CPU-GPU data transfers for maximum throughput.

        Args:
            traces: Input traces (n_samples, n_traces)
            aperture: Spatial aperture size
            sample_rate: Sample rate in Hz

        Returns:
            Denoised traces
        """
        n_samples, n_traces = traces.shape
        half_aperture = aperture // 2

        start_time = time.time()

        # ===== STEP 1: BULK FORWARD STFT =====
        # Compute STFT for ALL traces at once (single CPU→GPU transfer)
        t0 = time.time()
        stft_all, freqs = self.gpu_stft.batch_forward(
            traces,
            fmin=self.fmin,
            fmax=self.fmax,
            sample_rate=sample_rate
        )
        # stft_all shape: (n_traces, n_freqs, n_frames)
        time_forward = time.time() - t0

        n_freqs = stft_all.shape[1]
        n_frames = stft_all.shape[2]
        logger.debug(
            f"STFT bulk forward: {n_traces} traces → ({n_traces}, {n_freqs}, {n_frames}) in {time_forward:.3f}s"
        )

        # ===== STEP 2: SLIDING WINDOW THRESHOLDING =====
        # Apply MAD thresholding with spatial aperture
        t0 = time.time()
        denoised_stft = self._apply_sliding_window_threshold_stft(
            stft_all, half_aperture
        )
        time_threshold = time.time() - t0

        # ===== STEP 3: BULK INVERSE STFT =====
        # Compute inverse STFT for ALL traces at once (single GPU→CPU transfer)
        t0 = time.time()
        denoised_traces = self.gpu_stft.batch_inverse(
            denoised_stft,
            signal_length=n_samples
        )
        time_inverse = time.time() - t0

        elapsed_total = time.time() - start_time

        # Log timing breakdown for bottleneck identification
        logger.debug(
            f"STFT bulk timing: Forward={time_forward:.2f}s ({time_forward/elapsed_total*100:.0f}%) | "
            f"Threshold={time_threshold:.2f}s ({time_threshold/elapsed_total*100:.0f}%) | "
            f"Inverse={time_inverse:.2f}s ({time_inverse/elapsed_total*100:.0f}%)"
        )

        return denoised_traces

    def _apply_sliding_window_threshold_stft(
        self,
        stft_all: np.ndarray,
        half_aperture: int
    ) -> np.ndarray:
        """
        Apply sliding window MAD thresholding to pre-computed STFT coefficients.

        FULLY GPU-RESIDENT: Transfers data to GPU once, processes all traces,
        and transfers back once. Minimizes CPU-GPU transfers.

        Args:
            stft_all: STFT coefficients (n_traces, n_freqs, n_frames)
            half_aperture: Half of spatial aperture size

        Returns:
            Denoised STFT coefficients (n_traces, n_freqs, n_frames)
        """
        n_traces, n_freqs, n_frames = stft_all.shape

        # Transfer ALL data to GPU ONCE
        stft_gpu = torch.from_numpy(stft_all).to(self.device)
        denoised_gpu = torch.zeros_like(stft_gpu)

        # Pre-compute magnitudes and phases on GPU
        magnitudes_all = torch.abs(stft_gpu)  # (n_traces, n_freqs, n_frames)
        phases_all = torch.angle(stft_gpu)

        # Process each trace with its spatial aperture (all on GPU)
        for i in range(n_traces):
            start_idx = max(0, i - half_aperture)
            end_idx = min(n_traces, i + half_aperture + 1)
            center_idx = i - start_idx

            # Extract spatial aperture (GPU slice, no copy)
            magnitudes_ensemble = magnitudes_all[start_idx:end_idx]  # (aperture, n_freqs, n_frames)

            # Compute spatial statistics on GPU
            median_mag = torch.median(magnitudes_ensemble, dim=0, keepdim=True).values
            abs_dev = torch.abs(magnitudes_ensemble - median_mag)
            mad = torch.median(abs_dev, dim=0, keepdim=True).values * 1.4826

            # Prevent MAD=0
            min_mad = torch.maximum(0.01 * median_mag, torch.tensor(1e-10, device=self.device))
            mad = torch.maximum(mad, min_mad)

            # Outlier threshold
            outlier_threshold = self.threshold_k * mad

            # Get center trace magnitudes
            center_mag = magnitudes_all[i:i+1]  # Keep dim for broadcasting
            deviation = torch.abs(center_mag - median_mag)

            # Apply adaptive thresholding (most common mode)
            outlier_ratio = deviation / (outlier_threshold + 1e-10)
            severe_threshold = 2.0

            # Scaled removal for moderate outliers
            moderate_removal = torch.clamp(
                (outlier_ratio - 1.0) / (severe_threshold - 1.0), 0.0, 1.0
            )
            signs = torch.where(center_mag >= median_mag, 1.0, -1.0)
            moderate_new_deviation = deviation * (1.0 - moderate_removal)
            moderate_magnitude = torch.maximum(
                median_mag + signs * moderate_new_deviation,
                torch.zeros_like(center_mag)
            )

            new_magnitude = torch.where(
                outlier_ratio > severe_threshold,
                median_mag,  # Severe: full removal
                torch.where(
                    outlier_ratio > 1.0,
                    moderate_magnitude,  # Moderate: scaled
                    center_mag  # Non-outlier: keep original
                )
            )

            # Reconstruct complex value
            denoised_gpu[i] = (new_magnitude[0] * torch.exp(1j * phases_all[i]))

        # Transfer back to CPU ONCE
        return denoised_gpu.cpu().numpy()
