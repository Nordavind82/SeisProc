"""
3D FKK (Frequency-Wavenumber-Wavenumber) Filter Processor

GPU-accelerated 3D FKK filter for removing coherent noise from 3D seismic volumes.
Implements velocity cone filtering in the 3D Fourier domain.

Uses PyTorch for GPU acceleration with automatic CPU fallback.
"""
import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any
import logging

from models.seismic_volume import SeismicVolume
from models.fkk_config import FKKConfig
from processors.gpu.device_manager import DeviceManager, get_device_manager


logger = logging.getLogger(__name__)


class FKKFilterGPU:
    """
    3D FKK filter with GPU acceleration.

    Applies velocity cone filtering in the 3D Fourier domain (f, kx, ky).
    Velocity relationship: v = f / sqrt(kx² + ky²)
    """

    def __init__(self, device_manager: Optional[DeviceManager] = None):
        """
        Initialize FKK filter processor.

        Args:
            device_manager: DeviceManager instance for GPU handling.
                           If None, uses global device manager.
        """
        self.device_manager = device_manager or get_device_manager()
        self.device = self.device_manager.device

        # Cache for avoiding recomputation
        self._spectrum_cache: Optional[torch.Tensor] = None
        self._spectrum_shape: Optional[Tuple[int, int, int]] = None
        self._mask_cache: Optional[torch.Tensor] = None
        self._last_config: Optional[FKKConfig] = None

        logger.info(f"FKKFilterGPU initialized on {self.device}")

    def compute_spectrum(self, volume: SeismicVolume) -> np.ndarray:
        """
        Compute 3D FKK amplitude spectrum.

        Args:
            volume: Input SeismicVolume

        Returns:
            3D amplitude spectrum (nf, nkx, nky) as numpy array
            where nf = nt//2+1 (positive frequencies only)
        """
        logger.debug(f"Computing FKK spectrum for volume {volume.shape}")

        # Transfer to GPU
        data_tensor = torch.from_numpy(volume.data).to(self.device)

        # Apply rfft on time axis (0), fft on spatial axes (1, 2)
        # This gives output shape (nf, nx, ny) where nf = nt//2+1
        spectrum = torch.fft.rfft(data_tensor, dim=0)  # Time -> frequency
        spectrum = torch.fft.fft(spectrum, dim=1)       # X -> kx
        spectrum = torch.fft.fft(spectrum, dim=2)       # Y -> ky

        # Shift kx and ky axes to center (not frequency axis)
        spectrum = torch.fft.fftshift(spectrum, dim=(1, 2))

        # Cache the complex spectrum for filter application
        self._spectrum_cache = spectrum
        self._spectrum_shape = volume.shape

        # Return amplitude spectrum
        amplitude = torch.abs(spectrum)
        return amplitude.cpu().numpy()

    def compute_spectrum_slices(
        self,
        volume: SeismicVolume
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Compute FKK spectrum and return key slices for visualization.

        Args:
            volume: Input SeismicVolume

        Returns:
            Tuple of:
                - kxky_slice: Kx-Ky slice at mid frequency (nkx, nky)
                - fkx_slice: F-Kx slice at ky=0 (nf, nkx)
                - fky_slice: F-Ky slice at kx=0 (nf, nky)
                - axes: Dictionary with axis arrays
        """
        # Compute full spectrum
        spectrum = self.compute_spectrum(volume)
        nf, nkx, nky = spectrum.shape

        # Extract slices
        mid_f = nf // 2
        mid_kx = nkx // 2
        mid_ky = nky // 2

        kxky_slice = spectrum[mid_f, :, :]    # Kx-Ky at mid frequency
        fkx_slice = spectrum[:, :, mid_ky]    # F-Kx at ky=0
        fky_slice = spectrum[:, mid_kx, :]    # F-Ky at kx=0

        # Compute axes
        axes = self._compute_axes(volume)

        return kxky_slice, fkx_slice, fky_slice, axes

    def _compute_axes(self, volume: SeismicVolume) -> Dict[str, np.ndarray]:
        """
        Compute frequency and wavenumber axes.

        Args:
            volume: SeismicVolume with geometry

        Returns:
            Dictionary with f_axis, kx_axis, ky_axis
        """
        nt, nx, ny = volume.shape
        nf = nt // 2 + 1  # rfft output size

        # Frequency axis (positive only for rfft)
        f_axis = np.fft.rfftfreq(nt, volume.dt)

        # Wavenumber axes (centered)
        kx_axis = np.fft.fftshift(np.fft.fftfreq(nx, volume.dx))
        ky_axis = np.fft.fftshift(np.fft.fftfreq(ny, volume.dy))

        return {
            'f_axis': f_axis,
            'kx_axis': kx_axis,
            'ky_axis': ky_axis,
            'nf': nf,
            'nkx': nx,
            'nky': ny
        }

    def build_velocity_cone_mask(
        self,
        shape: Tuple[int, int, int],
        geometry: Tuple[float, float, float],
        config: FKKConfig
    ) -> np.ndarray:
        """
        Build 3D velocity cone filter mask on GPU.

        Args:
            shape: Volume shape (nt, nx, ny)
            geometry: (dt, dx, dy) in (seconds, meters, meters)
            config: FKKConfig with filter parameters

        Returns:
            3D filter mask (nf, nkx, nky) with values 0-1
        """
        nt, nx, ny = shape
        dt, dx, dy = geometry
        nf = nt // 2 + 1  # rfft output size

        logger.debug(f"Building velocity cone mask: shape=({nf}, {nx}, {ny}), "
                     f"v_range=[{config.v_min}, {config.v_max}] m/s")

        # Compute axes on GPU
        f_axis = torch.fft.rfftfreq(nt, dt, device=self.device)
        kx_axis = torch.fft.fftshift(torch.fft.fftfreq(nx, dx, device=self.device))
        ky_axis = torch.fft.fftshift(torch.fft.fftfreq(ny, dy, device=self.device))

        # Create 3D grids
        f_grid, kx_grid, ky_grid = torch.meshgrid(f_axis, kx_axis, ky_axis, indexing='ij')

        # Compute horizontal wavenumber magnitude
        k_horizontal = torch.sqrt(kx_grid**2 + ky_grid**2)

        # Avoid division by zero
        k_horizontal = torch.clamp(k_horizontal, min=1e-10)

        # Compute velocity at each point: v = f / k_horizontal
        velocity = torch.abs(f_grid) / k_horizontal

        # Compute azimuth (degrees, 0-360)
        azimuth = torch.rad2deg(torch.atan2(ky_grid, kx_grid)) % 360.0

        # Build mask based on velocity and azimuth
        v_min, v_max = config.v_min, config.v_max
        az_min, az_max = config.azimuth_min, config.azimuth_max

        # Velocity condition
        in_velocity_range = (velocity >= v_min) & (velocity <= v_max)

        # Azimuth condition (handle wrap-around)
        if az_min <= az_max:
            in_azimuth_range = (azimuth >= az_min) & (azimuth <= az_max)
        else:
            # Wrap around case (e.g., 350° to 10°)
            in_azimuth_range = (azimuth >= az_min) | (azimuth <= az_max)

        # Combined condition
        in_filter_zone = in_velocity_range & in_azimuth_range

        # Create mask based on mode
        if config.mode == 'reject':
            # Reject: set filter zone to 0
            mask = torch.ones_like(velocity)
            mask[in_filter_zone] = 0.0
        else:
            # Pass: keep only filter zone
            mask = torch.zeros_like(velocity)
            mask[in_filter_zone] = 1.0

        # Apply taper at velocity boundaries
        mask = self._apply_velocity_taper(
            mask, velocity, v_min, v_max, config.taper_width, config.mode
        )

        # Preserve DC component (f=0, kx=0, ky=0)
        mid_kx = nx // 2
        mid_ky = ny // 2
        mask[0, mid_kx, mid_ky] = 1.0

        # Cache mask
        self._mask_cache = mask
        self._last_config = config.copy()

        return mask.cpu().numpy()

    def _apply_velocity_taper(
        self,
        mask: torch.Tensor,
        velocity: torch.Tensor,
        v_min: float,
        v_max: float,
        taper_width: float,
        mode: str
    ) -> torch.Tensor:
        """
        Apply cosine taper at velocity boundaries.

        Args:
            mask: Current mask tensor
            velocity: Velocity at each point
            v_min, v_max: Velocity boundaries
            taper_width: Taper width as fraction of boundary
            mode: 'reject' or 'pass'

        Returns:
            Tapered mask
        """
        if taper_width <= 0:
            return mask

        # Taper widths in m/s
        taper_low = v_min * taper_width
        taper_high = v_max * taper_width

        if mode == 'reject':
            # Taper from 1 to 0 as we enter the reject zone

            # Lower boundary taper (v approaching v_min from below)
            v1 = v_min - taper_low
            v2 = v_min + taper_low
            in_lower_taper = (velocity >= v1) & (velocity < v2)
            if in_lower_taper.any():
                taper_val = 0.5 * (1.0 + torch.cos(
                    torch.pi * (velocity[in_lower_taper] - v1) / (2.0 * taper_low)
                ))
                mask[in_lower_taper] = taper_val

            # Upper boundary taper (v approaching v_max from above)
            v3 = v_max - taper_high
            v4 = v_max + taper_high
            in_upper_taper = (velocity > v3) & (velocity <= v4)
            if in_upper_taper.any():
                taper_val = 0.5 * (1.0 - torch.cos(
                    torch.pi * (velocity[in_upper_taper] - v3) / (2.0 * taper_high)
                ))
                mask[in_upper_taper] = taper_val

        else:  # mode == 'pass'
            # Taper from 0 to 1 as we enter the pass zone

            # Lower boundary taper
            v1 = v_min - taper_low
            v2 = v_min + taper_low
            in_lower_taper = (velocity >= v1) & (velocity < v2)
            if in_lower_taper.any():
                taper_val = 0.5 * (1.0 - torch.cos(
                    torch.pi * (velocity[in_lower_taper] - v1) / (2.0 * taper_low)
                ))
                mask[in_lower_taper] = taper_val

            # Upper boundary taper
            v3 = v_max - taper_high
            v4 = v_max + taper_high
            in_upper_taper = (velocity > v3) & (velocity <= v4)
            if in_upper_taper.any():
                taper_val = 0.5 * (1.0 + torch.cos(
                    torch.pi * (velocity[in_upper_taper] - v3) / (2.0 * taper_high)
                ))
                mask[in_upper_taper] = taper_val

        return mask

    def apply_filter(
        self,
        volume: SeismicVolume,
        config: FKKConfig,
        use_cached_spectrum: bool = True
    ) -> SeismicVolume:
        """
        Apply FKK filter to volume.

        Args:
            volume: Input SeismicVolume
            config: FKKConfig with filter parameters
            use_cached_spectrum: If True, reuse cached spectrum if available

        Returns:
            Filtered SeismicVolume
        """
        logger.info(f"Applying FKK filter: {config.get_summary()}")

        # Check if we can use cached spectrum
        use_cache = (
            use_cached_spectrum and
            self._spectrum_cache is not None and
            self._spectrum_shape == volume.shape
        )

        if use_cache:
            logger.debug("Using cached spectrum")
            spectrum = self._spectrum_cache
        else:
            # Compute spectrum: rfft on time, fft on spatial
            data_tensor = torch.from_numpy(volume.data).to(self.device)
            spectrum = torch.fft.rfft(data_tensor, dim=0)
            spectrum = torch.fft.fft(spectrum, dim=1)
            spectrum = torch.fft.fft(spectrum, dim=2)
            spectrum = torch.fft.fftshift(spectrum, dim=(1, 2))

        # Build filter mask
        mask = self.build_velocity_cone_mask(
            volume.shape,
            (volume.dt, volume.dx, volume.dy),
            config
        )
        mask_tensor = torch.from_numpy(mask).to(self.device)

        # Apply filter
        filtered_spectrum = spectrum * mask_tensor

        # Inverse FFT: ifft on spatial, irfft on time
        filtered_spectrum = torch.fft.ifftshift(filtered_spectrum, dim=(1, 2))
        filtered_spectrum = torch.fft.ifft(filtered_spectrum, dim=2)
        filtered_spectrum = torch.fft.ifft(filtered_spectrum, dim=1)
        filtered_data = torch.fft.irfft(filtered_spectrum, n=volume.nt, dim=0)

        # Create result volume
        result = SeismicVolume(
            data=filtered_data.real.cpu().numpy().astype(np.float32),
            dt=volume.dt,
            dx=volume.dx,
            dy=volume.dy,
            metadata={
                **volume.metadata,
                'fkk_filter_applied': True,
                'fkk_config': config.to_dict()
            }
        )

        logger.info("FKK filter applied successfully")
        return result

    def get_mask_slices(
        self,
        volume: SeismicVolume,
        config: FKKConfig
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get filter mask slices for visualization.

        Args:
            volume: SeismicVolume for geometry
            config: FKKConfig

        Returns:
            Tuple of (kxky_mask, fkx_mask, fky_mask)
        """
        mask = self.build_velocity_cone_mask(
            volume.shape,
            (volume.dt, volume.dx, volume.dy),
            config
        )

        nf, nkx, nky = mask.shape
        mid_f = nf // 2
        mid_kx = nkx // 2
        mid_ky = nky // 2

        kxky_mask = mask[mid_f, :, :]
        fkx_mask = mask[:, :, mid_ky]
        fky_mask = mask[:, mid_kx, :]

        return kxky_mask, fkx_mask, fky_mask

    def clear_cache(self):
        """Clear GPU memory cache."""
        self._spectrum_cache = None
        self._spectrum_shape = None
        self._mask_cache = None
        self._last_config = None

        self.device_manager.clear_cache()
        logger.debug("FKK filter cache cleared")

    def estimate_memory_mb(self, shape: Tuple[int, int, int]) -> float:
        """
        Estimate GPU memory required for processing.

        Args:
            shape: Volume shape (nt, nx, ny)

        Returns:
            Estimated memory in MB
        """
        nt, nx, ny = shape
        nf = nt // 2 + 1

        # Memory components (bytes)
        input_size = nt * nx * ny * 4           # float32 input
        spectrum_size = nf * nx * ny * 8        # complex64 spectrum
        mask_size = nf * nx * ny * 4            # float32 mask
        output_size = nt * nx * ny * 4          # float32 output

        total_bytes = input_size + spectrum_size + mask_size + output_size

        # Add 20% overhead for intermediates
        total_bytes *= 1.2

        return total_bytes / (1024 * 1024)

    def can_process_volume(self, volume: SeismicVolume) -> bool:
        """
        Check if volume can be processed with available GPU memory.

        Args:
            volume: SeismicVolume to check

        Returns:
            True if volume fits in GPU memory
        """
        required_mb = self.estimate_memory_mb(volume.shape)
        mem_info = self.device_manager.get_memory_info()

        if mem_info['available'] is None:
            # MPS or CPU - use conservative limit
            return required_mb < 4000  # 4GB limit

        available_mb = mem_info['available'] / (1024 * 1024)
        # Use 80% of available memory
        return required_mb < (available_mb * 0.8)

    def get_status(self) -> Dict[str, Any]:
        """
        Get processor status for UI display.

        Returns:
            Dictionary with status information
        """
        mem_info = self.device_manager.get_memory_info()

        return {
            'device': str(self.device),
            'device_name': self.device_manager.get_device_name(),
            'gpu_available': self.device_manager.is_gpu_available(),
            'spectrum_cached': self._spectrum_cache is not None,
            'cached_shape': self._spectrum_shape,
            'memory_allocated_mb': (
                mem_info['allocated'] / (1024 * 1024)
                if mem_info['allocated'] else None
            ),
            'memory_total_mb': (
                mem_info['total'] / (1024 * 1024)
                if mem_info['total'] else None
            ),
        }


class FKKFilterCPU:
    """
    CPU fallback implementation of FKK filter.

    Uses scipy.fft for systems without GPU support.
    Same interface as FKKFilterGPU.
    """

    def __init__(self):
        """Initialize CPU-based FKK filter."""
        from scipy import fft
        self._fft = fft
        self._spectrum_cache: Optional[np.ndarray] = None
        self._spectrum_shape: Optional[Tuple[int, int, int]] = None

        logger.info("FKKFilterCPU initialized (CPU fallback mode)")

    def compute_spectrum(self, volume: SeismicVolume) -> np.ndarray:
        """Compute 3D FKK spectrum on CPU."""
        # Apply rfft on time axis (0), fft on spatial axes (1, 2)
        spectrum = self._fft.rfft(volume.data, axis=0)
        spectrum = self._fft.fft(spectrum, axis=1)
        spectrum = self._fft.fft(spectrum, axis=2)
        spectrum = self._fft.fftshift(spectrum, axes=(1, 2))

        self._spectrum_cache = spectrum
        self._spectrum_shape = volume.shape

        return np.abs(spectrum)

    def build_velocity_cone_mask(
        self,
        shape: Tuple[int, int, int],
        geometry: Tuple[float, float, float],
        config: FKKConfig
    ) -> np.ndarray:
        """Build velocity cone mask on CPU."""
        nt, nx, ny = shape
        dt, dx, dy = geometry
        nf = nt // 2 + 1

        # Compute axes
        f_axis = np.fft.rfftfreq(nt, dt)
        kx_axis = np.fft.fftshift(np.fft.fftfreq(nx, dx))
        ky_axis = np.fft.fftshift(np.fft.fftfreq(ny, dy))

        # Create grids
        f_grid, kx_grid, ky_grid = np.meshgrid(f_axis, kx_axis, ky_axis, indexing='ij')

        # Compute velocity
        k_horizontal = np.sqrt(kx_grid**2 + ky_grid**2)
        k_horizontal = np.maximum(k_horizontal, 1e-10)
        velocity = np.abs(f_grid) / k_horizontal

        # Build mask
        v_min, v_max = config.v_min, config.v_max
        in_velocity_range = (velocity >= v_min) & (velocity <= v_max)

        if config.mode == 'reject':
            mask = np.ones_like(velocity, dtype=np.float32)
            mask[in_velocity_range] = 0.0
        else:
            mask = np.zeros_like(velocity, dtype=np.float32)
            mask[in_velocity_range] = 1.0

        # Preserve DC
        mid_kx = nx // 2
        mid_ky = ny // 2
        mask[0, mid_kx, mid_ky] = 1.0

        return mask

    def apply_filter(
        self,
        volume: SeismicVolume,
        config: FKKConfig,
        use_cached_spectrum: bool = True
    ) -> SeismicVolume:
        """Apply FKK filter on CPU."""
        # Compute or use cached spectrum
        if use_cached_spectrum and self._spectrum_cache is not None:
            spectrum = self._spectrum_cache
        else:
            # rfft on time, fft on spatial
            spectrum = self._fft.rfft(volume.data, axis=0)
            spectrum = self._fft.fft(spectrum, axis=1)
            spectrum = self._fft.fft(spectrum, axis=2)
            spectrum = self._fft.fftshift(spectrum, axes=(1, 2))

        # Build mask
        mask = self.build_velocity_cone_mask(
            volume.shape,
            (volume.dt, volume.dx, volume.dy),
            config
        )

        # Apply filter
        filtered_spectrum = spectrum * mask

        # Inverse FFT: ifft on spatial, irfft on time
        filtered_spectrum = self._fft.ifftshift(filtered_spectrum, axes=(1, 2))
        filtered_spectrum = self._fft.ifft(filtered_spectrum, axis=2)
        filtered_spectrum = self._fft.ifft(filtered_spectrum, axis=1)
        filtered_data = self._fft.irfft(filtered_spectrum, n=volume.nt, axis=0)

        return SeismicVolume(
            data=filtered_data.real.astype(np.float32),
            dt=volume.dt,
            dx=volume.dx,
            dy=volume.dy,
            metadata={
                **volume.metadata,
                'fkk_filter_applied': True,
                'fkk_config': config.to_dict()
            }
        )

    def clear_cache(self):
        """Clear spectrum cache."""
        self._spectrum_cache = None
        self._spectrum_shape = None


def get_fkk_filter(prefer_gpu: bool = True) -> 'FKKFilterGPU | FKKFilterCPU':
    """
    Get appropriate FKK filter based on hardware availability.

    Args:
        prefer_gpu: If True, use GPU if available

    Returns:
        FKKFilterGPU if GPU available and preferred, else FKKFilterCPU
    """
    if prefer_gpu:
        try:
            device_manager = get_device_manager()
            if device_manager.is_gpu_available():
                return FKKFilterGPU(device_manager)
        except Exception as e:
            logger.warning(f"GPU initialization failed, falling back to CPU: {e}")

    return FKKFilterCPU()
