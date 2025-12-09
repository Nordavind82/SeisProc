"""
3D FKK (Frequency-Wavenumber-Wavenumber) Filter Processor

GPU-accelerated 3D FKK filter for removing coherent noise from 3D seismic volumes.
Implements velocity cone filtering in the 3D Fourier domain.

Simple single-pass architecture:
1. Pad volume to power-of-2 for FFT efficiency
2. 3D FFT
3. Apply velocity cone mask
4. 3D IFFT
5. Remove padding
"""
import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any
import logging

from models.seismic_volume import SeismicVolume
from models.fkk_config import FKKConfig
from processors.gpu.device_manager import DeviceManager, get_device_manager
from processors.agc import apply_agc_vectorized

logger = logging.getLogger(__name__)


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    power = 1
    while power < n:
        power *= 2
    return power


def apply_agc_3d(data: np.ndarray, window_samples: int, max_gain: float = 10.0
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Apply AGC to 3D volume trace-by-trace."""
    nt, nx, ny = data.shape
    data_2d = data.reshape(nt, -1)
    agc_data, scale_factors = apply_agc_vectorized(
        data_2d, window_samples=window_samples, target_rms=1.0, max_gain=max_gain
    )
    return agc_data.reshape(nt, nx, ny), scale_factors.reshape(nt, nx, ny)


def remove_agc_3d(data: np.ndarray, scale_factors: np.ndarray) -> np.ndarray:
    """Remove AGC from 3D volume using stored scale factors."""
    safe_scales = np.where(scale_factors > 1e-10, scale_factors, 1.0)
    return data / safe_scales


def apply_temporal_taper(data: np.ndarray, taper_top: int, taper_bottom: int) -> np.ndarray:
    """Apply cosine taper to top and bottom of traces."""
    if taper_top == 0 and taper_bottom == 0:
        return data

    nt = data.shape[0]
    result = data.copy()

    if taper_top > 0 and taper_top < nt:
        taper = 0.5 * (1 - np.cos(np.pi * np.arange(taper_top) / taper_top))
        result[:taper_top] *= taper[:, np.newaxis, np.newaxis]

    if taper_bottom > 0 and taper_bottom < nt:
        taper = 0.5 * (1 + np.cos(np.pi * np.arange(taper_bottom) / taper_bottom))
        result[-taper_bottom:] *= taper[:, np.newaxis, np.newaxis]

    return result


class FKKFilterGPU:
    """
    3D FKK filter with GPU acceleration.

    Simple single-pass implementation:
    - Pads to power-of-2 for FFT efficiency
    - Applies velocity cone mask in Fourier domain
    - No block processing needed
    """

    def __init__(self, device_manager: Optional[DeviceManager] = None):
        """Initialize FKK filter processor."""
        self.device_manager = device_manager or get_device_manager()
        self.device = self.device_manager.device
        self._spectrum_cache = None
        logger.info(f"FKKFilterGPU initialized on {self.device}")

    def build_velocity_cone_mask(
        self,
        shape: Tuple[int, int, int],
        geometry: Tuple[float, float, float],
        config: FKKConfig
    ) -> torch.Tensor:
        """
        Build 3D velocity cone filter mask on GPU.

        Args:
            shape: Volume shape (nt, nx, ny)
            geometry: (dt, dx, dy) in seconds and meters
            config: Filter configuration

        Returns:
            Filter mask tensor on GPU, shape (nf, nx, ny)
        """
        nt, nx, ny = shape
        dt, dx, dy = geometry

        nyquist = 0.5 / dt
        f_min = config.f_min if config.f_min is not None else 0.0
        f_max = config.f_max if config.f_max is not None else nyquist

        # Compute frequency and wavenumber axes
        f_axis = torch.fft.rfftfreq(nt, dt, device=self.device)
        kx_axis = torch.fft.fftshift(torch.fft.fftfreq(nx, dx, device=self.device))
        ky_axis = torch.fft.fftshift(torch.fft.fftfreq(ny, dy, device=self.device))

        # Create 3D grids
        f_grid, kx_grid, ky_grid = torch.meshgrid(f_axis, kx_axis, ky_axis, indexing='ij')

        # Compute horizontal wavenumber magnitude
        k_horizontal = torch.sqrt(kx_grid**2 + ky_grid**2)

        # Compute apparent velocity: v = f / k_horizontal
        # Avoid division by zero - set k=0 points to very small value
        k_safe = torch.where(k_horizontal > 1e-10, k_horizontal, torch.tensor(1e-10, device=self.device))
        velocity = torch.abs(f_grid) / k_safe

        # Compute azimuth for directional filtering
        azimuth = torch.rad2deg(torch.atan2(ky_grid, kx_grid)) % 360.0

        # Determine which points are in filter zone
        v_min, v_max = config.v_min, config.v_max
        az_min, az_max = config.azimuth_min, config.azimuth_max

        in_frequency_band = (torch.abs(f_grid) >= f_min) & (torch.abs(f_grid) <= f_max)
        in_velocity_range = (velocity >= v_min) & (velocity <= v_max)

        if az_min <= az_max:
            in_azimuth_range = (azimuth >= az_min) & (azimuth <= az_max)
        else:
            in_azimuth_range = (azimuth >= az_min) | (azimuth <= az_max)

        in_filter_zone = in_velocity_range & in_azimuth_range & in_frequency_band

        # Build mask based on mode
        if config.mode == 'reject':
            mask = torch.ones_like(velocity)
            mask[in_filter_zone] = 0.0
        else:  # pass mode
            mask = torch.zeros_like(velocity)
            mask[in_filter_zone] = 1.0
            # Pass frequencies outside the filter band
            mask[~in_frequency_band] = 1.0

        # Apply velocity taper for smooth transition
        if config.taper_width > 0:
            mask = self._apply_velocity_taper(
                mask, velocity, v_min, v_max, config.taper_width, config.mode, in_frequency_band
            )

        # CRITICAL: Preserve DC and zero-frequency components
        # f=0 row: velocity is undefined (0/k = 0), preserve all
        mask[0, :, :] = 1.0
        # kx=ky=0 column: velocity is infinite (f/0), preserve all
        mask[:, nx // 2, ny // 2] = 1.0

        return mask

    def _apply_velocity_taper(
        self,
        mask: torch.Tensor,
        velocity: torch.Tensor,
        v_min: float,
        v_max: float,
        taper_width: float,
        mode: str,
        in_frequency_band: torch.Tensor
    ) -> torch.Tensor:
        """Apply cosine taper at velocity boundaries."""
        taper_low = v_min * taper_width
        taper_high = v_max * taper_width

        if mode == 'reject':
            # Taper from 1 to 0 at lower boundary
            v1, v2 = v_min - taper_low, v_min + taper_low
            in_lower = (velocity >= v1) & (velocity < v2) & in_frequency_band
            if in_lower.any():
                t = (velocity[in_lower] - v1) / (2.0 * taper_low)
                mask[in_lower] = 0.5 * (1.0 + torch.cos(torch.pi * t))

            # Taper from 0 to 1 at upper boundary
            v3, v4 = v_max - taper_high, v_max + taper_high
            in_upper = (velocity > v3) & (velocity <= v4) & in_frequency_band
            if in_upper.any():
                t = (velocity[in_upper] - v3) / (2.0 * taper_high)
                mask[in_upper] = 0.5 * (1.0 - torch.cos(torch.pi * t))
        else:  # pass mode
            # Taper from 0 to 1 at lower boundary
            v1, v2 = v_min - taper_low, v_min + taper_low
            in_lower = (velocity >= v1) & (velocity < v2) & in_frequency_band
            if in_lower.any():
                t = (velocity[in_lower] - v1) / (2.0 * taper_low)
                mask[in_lower] = 0.5 * (1.0 - torch.cos(torch.pi * t))

            # Taper from 1 to 0 at upper boundary
            v3, v4 = v_max - taper_high, v_max + taper_high
            in_upper = (velocity > v3) & (velocity <= v4) & in_frequency_band
            if in_upper.any():
                t = (velocity[in_upper] - v3) / (2.0 * taper_high)
                mask[in_upper] = 0.5 * (1.0 + torch.cos(torch.pi * t))

        return mask

    def apply_filter(self, volume: SeismicVolume, config: FKKConfig) -> SeismicVolume:
        """
        Apply FKK filter to seismic volume.

        Args:
            volume: Input SeismicVolume
            config: Filter configuration

        Returns:
            Filtered SeismicVolume
        """
        logger.info(f"Applying FKK filter: {config.get_summary()}")

        nt, nx, ny = volume.shape
        data = volume.data.astype(np.float32).copy()
        agc_scale_factors = None

        # Step 1: Optional AGC
        if config.apply_agc:
            window_samples = max(3, int(config.agc_window_ms / 1000.0 / volume.dt))
            data, agc_scale_factors = apply_agc_3d(data, window_samples, config.agc_max_gain)
            logger.debug(f"Applied AGC with window={window_samples} samples")

        # Step 2: Optional temporal taper
        taper_top = int(config.taper_ms_top / 1000.0 / volume.dt) if config.taper_ms_top > 0 else 0
        taper_bottom = int(config.taper_ms_bottom / 1000.0 / volume.dt) if config.taper_ms_bottom > 0 else 0
        if taper_top > 0 or taper_bottom > 0:
            data = apply_temporal_taper(data, taper_top, taper_bottom)

        # Step 3: Pad to power-of-2 for FFT efficiency
        nt_pad = next_power_of_2(nt)
        nx_pad = next_power_of_2(nx)
        ny_pad = next_power_of_2(ny)

        pad_t = nt_pad - nt
        pad_x = nx_pad - nx
        pad_y = ny_pad - ny

        if pad_t > 0 or pad_x > 0 or pad_y > 0:
            data = np.pad(data, ((0, pad_t), (0, pad_x), (0, pad_y)), mode='constant')
            logger.debug(f"Padded: ({nt},{nx},{ny}) -> ({nt_pad},{nx_pad},{ny_pad})")

        # Step 4: Transfer to GPU and compute 3D FFT
        data_gpu = torch.from_numpy(data).to(self.device)

        spectrum = torch.fft.rfft(data_gpu, dim=0)
        spectrum = torch.fft.fft(spectrum, dim=1)
        spectrum = torch.fft.fft(spectrum, dim=2)
        spectrum = torch.fft.fftshift(spectrum, dim=(1, 2))

        # Step 5: Build and apply mask
        mask = self.build_velocity_cone_mask(
            (nt_pad, nx_pad, ny_pad),
            (volume.dt, volume.dx, volume.dy),
            config
        )
        spectrum = spectrum * mask

        # Step 6: Inverse 3D FFT
        spectrum = torch.fft.ifftshift(spectrum, dim=(1, 2))
        spectrum = torch.fft.ifft(spectrum, dim=2)
        spectrum = torch.fft.ifft(spectrum, dim=1)
        result = torch.fft.irfft(spectrum, n=nt_pad, dim=0)

        # Step 7: Transfer back to CPU and remove padding
        output = result.real.cpu().numpy().astype(np.float32)
        output = output[:nt, :nx, :ny]

        # Step 8: Remove AGC if applied
        if config.apply_agc and agc_scale_factors is not None:
            output = remove_agc_3d(output, agc_scale_factors)

        logger.info("FKK filter applied successfully")

        return SeismicVolume(
            data=output,
            dt=volume.dt,
            dx=volume.dx,
            dy=volume.dy,
            metadata={**volume.metadata, 'fkk_filter_applied': True}
        )

    def compute_spectrum(self, volume: SeismicVolume) -> np.ndarray:
        """Compute 3D FKK amplitude spectrum for visualization."""
        data_gpu = torch.from_numpy(volume.data).to(self.device)
        spectrum = torch.fft.rfft(data_gpu, dim=0)
        spectrum = torch.fft.fft(spectrum, dim=1)
        spectrum = torch.fft.fft(spectrum, dim=2)
        spectrum = torch.fft.fftshift(spectrum, dim=(1, 2))
        self._spectrum_cache = spectrum
        return torch.abs(spectrum).cpu().numpy()

    def _compute_axes(self, volume: SeismicVolume) -> Dict[str, np.ndarray]:
        """Compute frequency and wavenumber axes for visualization."""
        nt, nx, ny = volume.shape
        return {
            'f_axis': np.fft.rfftfreq(nt, volume.dt),
            'kx_axis': np.fft.fftshift(np.fft.fftfreq(nx, volume.dx)),
            'ky_axis': np.fft.fftshift(np.fft.fftfreq(ny, volume.dy)),
            'nf': nt // 2 + 1,
            'nkx': nx,
            'nky': ny
        }

    def get_mask_slices(self, volume: SeismicVolume, config: FKKConfig
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get filter mask slices for visualization."""
        mask = self.build_velocity_cone_mask(
            volume.shape, (volume.dt, volume.dx, volume.dy), config
        ).cpu().numpy()

        nf, nkx, nky = mask.shape
        mid_f, mid_kx, mid_ky = nf // 2, nkx // 2, nky // 2

        return mask[mid_f, :, :], mask[:, :, mid_ky], mask[:, mid_kx, :]

    def clear_cache(self):
        """Clear cached data."""
        self._spectrum_cache = None
        self.device_manager.clear_cache()

    def get_status(self) -> Dict[str, Any]:
        """Get processor status."""
        return {
            'device': str(self.device),
            'device_name': self.device_manager.get_device_name(),
            'gpu_available': self.device_manager.is_gpu_available(),
        }


class FKKFilterCPU:
    """CPU fallback implementation of FKK filter."""

    def __init__(self):
        from scipy import fft
        self._fft = fft
        self._spectrum_cache = None
        logger.info("FKKFilterCPU initialized")

    def build_velocity_cone_mask(
        self,
        shape: Tuple[int, int, int],
        geometry: Tuple[float, float, float],
        config: FKKConfig
    ) -> np.ndarray:
        """Build velocity cone mask on CPU."""
        nt, nx, ny = shape
        dt, dx, dy = geometry

        nyquist = 0.5 / dt
        f_min = config.f_min if config.f_min is not None else 0.0
        f_max = config.f_max if config.f_max is not None else nyquist

        f_axis = np.fft.rfftfreq(nt, dt)
        kx_axis = np.fft.fftshift(np.fft.fftfreq(nx, dx))
        ky_axis = np.fft.fftshift(np.fft.fftfreq(ny, dy))

        f_grid, kx_grid, ky_grid = np.meshgrid(f_axis, kx_axis, ky_axis, indexing='ij')

        k_horizontal = np.sqrt(kx_grid**2 + ky_grid**2)
        k_safe = np.where(k_horizontal > 1e-10, k_horizontal, 1e-10)
        velocity = np.abs(f_grid) / k_safe

        in_frequency_band = (np.abs(f_grid) >= f_min) & (np.abs(f_grid) <= f_max)
        in_velocity_range = (velocity >= config.v_min) & (velocity <= config.v_max)
        in_filter_zone = in_velocity_range & in_frequency_band

        if config.mode == 'reject':
            mask = np.ones_like(velocity, dtype=np.float32)
            mask[in_filter_zone] = 0.0
        else:
            mask = np.zeros_like(velocity, dtype=np.float32)
            mask[in_filter_zone] = 1.0
            mask[~in_frequency_band] = 1.0

        # Preserve DC components
        mask[0, :, :] = 1.0
        mask[:, nx // 2, ny // 2] = 1.0

        return mask

    def apply_filter(self, volume: SeismicVolume, config: FKKConfig) -> SeismicVolume:
        """Apply FKK filter on CPU."""
        logger.info(f"Applying FKK filter (CPU): {config.get_summary()}")

        nt, nx, ny = volume.shape
        data = volume.data.astype(np.float32).copy()
        agc_scale_factors = None

        if config.apply_agc:
            window_samples = max(3, int(config.agc_window_ms / 1000.0 / volume.dt))
            data, agc_scale_factors = apply_agc_3d(data, window_samples, config.agc_max_gain)

        taper_top = int(config.taper_ms_top / 1000.0 / volume.dt) if config.taper_ms_top > 0 else 0
        taper_bottom = int(config.taper_ms_bottom / 1000.0 / volume.dt) if config.taper_ms_bottom > 0 else 0
        if taper_top > 0 or taper_bottom > 0:
            data = apply_temporal_taper(data, taper_top, taper_bottom)

        nt_pad = next_power_of_2(nt)
        nx_pad = next_power_of_2(nx)
        ny_pad = next_power_of_2(ny)

        pad_t, pad_x, pad_y = nt_pad - nt, nx_pad - nx, ny_pad - ny
        if pad_t > 0 or pad_x > 0 or pad_y > 0:
            data = np.pad(data, ((0, pad_t), (0, pad_x), (0, pad_y)), mode='constant')

        spectrum = self._fft.rfft(data, axis=0)
        spectrum = self._fft.fft(spectrum, axis=1)
        spectrum = self._fft.fft(spectrum, axis=2)
        spectrum = self._fft.fftshift(spectrum, axes=(1, 2))

        mask = self.build_velocity_cone_mask(
            (nt_pad, nx_pad, ny_pad), (volume.dt, volume.dx, volume.dy), config
        )
        spectrum = spectrum * mask

        spectrum = self._fft.ifftshift(spectrum, axes=(1, 2))
        spectrum = self._fft.ifft(spectrum, axis=2)
        spectrum = self._fft.ifft(spectrum, axis=1)
        result = self._fft.irfft(spectrum, n=nt_pad, axis=0)

        output = result.real.astype(np.float32)[:nt, :nx, :ny]

        if config.apply_agc and agc_scale_factors is not None:
            output = remove_agc_3d(output, agc_scale_factors)

        return SeismicVolume(
            data=output, dt=volume.dt, dx=volume.dx, dy=volume.dy,
            metadata={**volume.metadata, 'fkk_filter_applied': True}
        )

    def compute_spectrum(self, volume: SeismicVolume) -> np.ndarray:
        """Compute spectrum on CPU."""
        spectrum = self._fft.rfft(volume.data, axis=0)
        spectrum = self._fft.fft(spectrum, axis=1)
        spectrum = self._fft.fft(spectrum, axis=2)
        spectrum = self._fft.fftshift(spectrum, axes=(1, 2))
        self._spectrum_cache = spectrum
        return np.abs(spectrum)

    def clear_cache(self):
        self._spectrum_cache = None


def get_fkk_filter(prefer_gpu: bool = True):
    """Get appropriate FKK filter based on hardware availability."""
    if prefer_gpu:
        try:
            device_manager = get_device_manager()
            if device_manager.is_gpu_available():
                return FKKFilterGPU(device_manager)
        except Exception as e:
            logger.warning(f"GPU init failed, using CPU: {e}")
    return FKKFilterCPU()
