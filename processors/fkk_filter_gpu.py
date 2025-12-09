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


def apply_agc_3d(data: np.ndarray, window_samples: int
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply AGC to 3D volume trace-by-trace.

    Uses adaptive epsilon (0.1% of global RMS) for natural gain limiting
    without creating shadow artifacts from outliers like air blasts.
    """
    nt, nx, ny = data.shape
    data_2d = data.reshape(nt, -1)
    agc_data, scale_factors = apply_agc_vectorized(
        data_2d, window_samples=window_samples, target_rms=1.0
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


# ===========================================================================
# Spatial Edge Handling - Pad-Copy Method
# ===========================================================================

def pad_copy_3d(data: np.ndarray, pad_x: int, pad_y: int) -> np.ndarray:
    """
    Pad volume with copies of edge traces, then apply taper to padded zone only.

    This method:
    1. Pads the volume with extra traces at edges
    2. Copies the edge trace into the padded region
    3. Applies cosine taper only to the padded traces (original data untouched)

    The copied traces create energy at k=0 (zero wavenumber), but since
    velocity is infinite at k=0 and we preserve k=0, this energy passes
    through unfiltered. After IFFT, we crop away the padded region.

    Args:
        data: 3D array (nt, nx, ny)
        pad_x: Number of traces to pad on each side in X
        pad_y: Number of traces to pad on each side in Y

    Returns:
        Padded array with shape (nt, nx + 2*pad_x, ny + 2*pad_y)
    """
    if pad_x == 0 and pad_y == 0:
        return data

    nt, nx, ny = data.shape

    # Create output array
    nx_out = nx + 2 * pad_x
    ny_out = ny + 2 * pad_y
    padded = np.zeros((nt, nx_out, ny_out), dtype=data.dtype)

    # Copy original data to center
    padded[:, pad_x:pad_x + nx, pad_y:pad_y + ny] = data

    # Create cosine taper for padded regions (1 at edge of data, 0 at far edge)
    if pad_x > 0:
        taper_x = 0.5 * (1 + np.cos(np.pi * np.arange(pad_x) / pad_x))  # 1 -> 0

        # Left padding: copy left edge trace and apply taper
        for i in range(pad_x):
            padded[:, pad_x - 1 - i, pad_y:pad_y + ny] = data[:, 0, :] * taper_x[i]

        # Right padding: copy right edge trace and apply taper
        for i in range(pad_x):
            padded[:, pad_x + nx + i, pad_y:pad_y + ny] = data[:, -1, :] * taper_x[i]

    if pad_y > 0:
        taper_y = 0.5 * (1 + np.cos(np.pi * np.arange(pad_y) / pad_y))  # 1 -> 0

        # Bottom padding (Y=0 side): copy bottom edge and apply taper
        for j in range(pad_y):
            padded[:, pad_x:pad_x + nx, pad_y - 1 - j] = data[:, :, 0] * taper_y[j]

        # Top padding (Y=ny side): copy top edge and apply taper
        for j in range(pad_y):
            padded[:, pad_x:pad_x + nx, pad_y + ny + j] = data[:, :, -1] * taper_y[j]

    # Handle corners (already zero, which is fine - they taper to zero from both directions)
    # Optionally could fill with tapered copies, but zero is simpler and works

    return padded


def get_auto_pad_size(dim: int, min_pad: int = 5, max_pad: int = 30) -> int:
    """
    Calculate automatic padding size based on dimension.

    Uses ~15% of dimension for small dims, ~10% for larger, clamped to range.

    Args:
        dim: Dimension size
        min_pad: Minimum padding (default 5)
        max_pad: Maximum padding (default 30)

    Returns:
        Padding size
    """
    # Use larger fraction for small dimensions (more prone to edge artifacts)
    if dim <= 20:
        auto_pad = max(min_pad, dim // 2)  # 50% for very small dims
    elif dim <= 50:
        auto_pad = max(min_pad, dim // 5)  # 20% for small dims
    else:
        auto_pad = max(min_pad, dim // 10)  # 10% for larger dims

    return min(auto_pad, max_pad)


def pad_copy_temporal(data: np.ndarray, pad_top: int, pad_bottom: int) -> np.ndarray:
    """
    Pad volume temporally with copies of edge samples, taper only padded zone.

    This reduces temporal edge artifacts from high-amplitude events at trace
    start/end by extending the data with copies of the first/last samples,
    then tapering only the padded zone to zero.

    The padded samples create f=0 (DC) energy which passes through the filter
    unchanged, while providing a smooth transition from zero to the data.

    Args:
        data: 3D numpy array (nt, nx, ny)
        pad_top: Number of samples to pad at top (before t=0)
        pad_bottom: Number of samples to pad at bottom (after t=nt-1)

    Returns:
        Padded 3D array (nt + pad_top + pad_bottom, nx, ny)
    """
    if pad_top == 0 and pad_bottom == 0:
        return data

    nt, nx, ny = data.shape
    nt_out = nt + pad_top + pad_bottom
    padded = np.zeros((nt_out, nx, ny), dtype=data.dtype)

    # Copy original data to center
    padded[pad_top:pad_top + nt, :, :] = data

    # Pad top with copies of first sample, apply taper
    if pad_top > 0:
        # Cosine taper: 1 at edge of data, 0 at far edge
        taper_top = 0.5 * (1 + np.cos(np.pi * np.arange(pad_top) / pad_top))  # 1 -> 0

        # First sample of original data (shape: nx, ny)
        first_sample = data[0, :, :]

        # Fill padded top region with tapered copies
        for t in range(pad_top):
            # t=0 is furthest from data (gets taper[pad_top-1] ≈ 0)
            # t=pad_top-1 is adjacent to data (gets taper[0] = 1)
            padded[pad_top - 1 - t, :, :] = first_sample * taper_top[t]

    # Pad bottom with copies of last sample, apply taper
    if pad_bottom > 0:
        # Cosine taper: 1 at edge of data, 0 at far edge
        taper_bottom = 0.5 * (1 + np.cos(np.pi * np.arange(pad_bottom) / pad_bottom))  # 1 -> 0

        # Last sample of original data (shape: nx, ny)
        last_sample = data[-1, :, :]

        # Fill padded bottom region with tapered copies
        for t in range(pad_bottom):
            # t=0 is adjacent to data (gets taper[0] = 1)
            # t=pad_bottom-1 is furthest (gets taper[pad_bottom-1] ≈ 0)
            padded[pad_top + nt + t, :, :] = last_sample * taper_bottom[t]

    return padded


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

        # Compute frequency and wavenumber axes (use float32 for MPS compatibility)
        f_axis = torch.fft.rfftfreq(nt, dt, device=self.device, dtype=torch.float32)
        kx_axis = torch.fft.fftshift(torch.fft.fftfreq(nx, dx, device=self.device, dtype=torch.float32))
        ky_axis = torch.fft.fftshift(torch.fft.fftfreq(ny, dy, device=self.device, dtype=torch.float32))

        # Create 3D grids
        f_grid, kx_grid, ky_grid = torch.meshgrid(f_axis, kx_axis, ky_axis, indexing='ij')

        # Compute horizontal wavenumber magnitude
        k_horizontal = torch.sqrt(kx_grid**2 + ky_grid**2)

        # Compute apparent velocity: v = f / k_horizontal
        # Avoid division by zero - set k=0 points to very small value
        k_safe = torch.where(k_horizontal > 1e-10, k_horizontal, torch.tensor(1e-10, device=self.device, dtype=torch.float32))
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
        logger.info(f"  Edge method: {config.edge_method}, pad_traces_x: {config.pad_traces_x}, pad_traces_y: {config.pad_traces_y}")
        logger.info(f"  Padding factor: {config.padding_factor}")
        print(f"[FKK] edge_method={config.edge_method}, pad_x={config.pad_traces_x}, pad_y={config.pad_traces_y}, fft_pad={config.padding_factor}")

        nt, nx, ny = volume.shape
        data = volume.data.astype(np.float32).copy()
        agc_scale_factors = None

        # Step 1: Optional AGC (uses adaptive epsilon, no max_gain clipping)
        if config.apply_agc:
            window_samples = max(3, int(config.agc_window_ms / 1000.0 / volume.dt))
            data, agc_scale_factors = apply_agc_3d(data, window_samples)
            logger.debug(f"Applied AGC with window={window_samples} samples")

        # Step 2: Optional temporal taper
        taper_top = int(config.taper_ms_top / 1000.0 / volume.dt) if config.taper_ms_top > 0 else 0
        taper_bottom = int(config.taper_ms_bottom / 1000.0 / volume.dt) if config.taper_ms_bottom > 0 else 0
        if taper_top > 0 or taper_bottom > 0:
            data = apply_temporal_taper(data, taper_top, taper_bottom)

        # Step 2b: Optional temporal pad_copy (for high-amplitude top/bottom edges)
        temporal_pad_top = int(config.pad_time_top_ms / 1000.0 / volume.dt) if config.pad_time_top_ms > 0 else 0
        temporal_pad_bottom = int(config.pad_time_bottom_ms / 1000.0 / volume.dt) if config.pad_time_bottom_ms > 0 else 0
        if temporal_pad_top > 0 or temporal_pad_bottom > 0:
            data = pad_copy_temporal(data, temporal_pad_top, temporal_pad_bottom)
            logger.info(f"  Applied temporal pad_copy: +{temporal_pad_top} samples at top, +{temporal_pad_bottom} at bottom")
            print(f"[FKK] temporal_pad: top={temporal_pad_top} samples, bottom={temporal_pad_bottom} samples")

        # Step 3: Spatial edge handling with pad_copy
        edge_pad_x = 0
        edge_pad_y = 0

        if config.edge_method == 'pad_copy':
            # Determine padding size (auto or user-specified)
            edge_pad_x = config.pad_traces_x if config.pad_traces_x > 0 else get_auto_pad_size(nx)
            edge_pad_y = config.pad_traces_y if config.pad_traces_y > 0 else get_auto_pad_size(ny)

            # Apply pad_copy: pad with copies of edge traces, taper only padded zone
            data = pad_copy_3d(data, edge_pad_x, edge_pad_y)
            logger.info(f"  Applied pad_copy: +{edge_pad_x} traces per side in X, +{edge_pad_y} traces per side in Y")

        # Step 4: Compute padded dimensions for FFT
        nt_current, nx_current, ny_current = data.shape

        # Apply padding factor for extra padding
        nt_pad = next_power_of_2(int(nt_current * config.padding_factor))
        nx_pad = next_power_of_2(int(nx_current * config.padding_factor))
        ny_pad = next_power_of_2(int(ny_current * config.padding_factor))

        # Ensure at least power-of-2 of current size
        nt_pad = max(nt_pad, next_power_of_2(nt_current))
        nx_pad = max(nx_pad, next_power_of_2(nx_current))
        ny_pad = max(ny_pad, next_power_of_2(ny_current))

        pad_t = nt_pad - nt_current
        pad_x = nx_pad - nx_current
        pad_y = ny_pad - ny_current

        if pad_t > 0 or pad_x > 0 or pad_y > 0:
            data = np.pad(data, ((0, pad_t), (0, pad_x), (0, pad_y)), mode='constant')
            logger.info(f"  FFT padded: ({nt_current},{nx_current},{ny_current}) -> ({nt_pad},{nx_pad},{ny_pad})")

        # Step 5: Transfer to GPU and compute 3D FFT
        # Use float32 for MPS compatibility (MPS doesn't support float64)
        data_gpu = torch.from_numpy(data).float().to(self.device)

        spectrum = torch.fft.rfft(data_gpu, dim=0)
        spectrum = torch.fft.fft(spectrum, dim=1)
        spectrum = torch.fft.fft(spectrum, dim=2)
        spectrum = torch.fft.fftshift(spectrum, dim=(1, 2))

        # Step 6: Build and apply mask
        mask = self.build_velocity_cone_mask(
            (nt_pad, nx_pad, ny_pad),
            (volume.dt, volume.dx, volume.dy),
            config
        )
        spectrum = spectrum * mask

        # Step 7: Inverse 3D FFT
        spectrum = torch.fft.ifftshift(spectrum, dim=(1, 2))
        spectrum = torch.fft.ifft(spectrum, dim=2)
        spectrum = torch.fft.ifft(spectrum, dim=1)
        result = torch.fft.irfft(spectrum, n=nt_pad, dim=0)

        # Step 8: Transfer back to CPU and remove FFT padding
        output = result.real.cpu().numpy().astype(np.float32)
        output = output[:nt_current, :nx_current, :ny_current]

        # Step 9: Remove temporal pad_copy (extract original time range)
        if temporal_pad_top > 0 or temporal_pad_bottom > 0:
            output = output[temporal_pad_top:temporal_pad_top + nt, :, :]

        # Step 9b: Remove spatial edge padding (extract original data region)
        if config.edge_method == 'pad_copy' and (edge_pad_x > 0 or edge_pad_y > 0):
            output = output[:, edge_pad_x:edge_pad_x + nx, edge_pad_y:edge_pad_y + ny]

        # Step 10: Remove AGC if applied
        if config.apply_agc and agc_scale_factors is not None:
            output = remove_agc_3d(output, agc_scale_factors)

        logger.info("FKK filter applied successfully")

        return SeismicVolume(
            data=output,
            dt=volume.dt,
            dx=volume.dx,
            dy=volume.dy,
            metadata={**volume.metadata, 'fkk_filter_applied': True, 'edge_method': config.edge_method}
        )

    def compute_spectrum(self, volume: SeismicVolume) -> np.ndarray:
        """Compute 3D FKK amplitude spectrum for visualization."""
        # Use float32 for MPS compatibility
        data_gpu = torch.from_numpy(volume.data).float().to(self.device)
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

        # Step 1: Optional AGC (uses adaptive epsilon, no max_gain clipping)
        if config.apply_agc:
            window_samples = max(3, int(config.agc_window_ms / 1000.0 / volume.dt))
            data, agc_scale_factors = apply_agc_3d(data, window_samples)
            logger.debug(f"Applied AGC with window={window_samples} samples")

        # Step 2: Optional temporal taper
        taper_top = int(config.taper_ms_top / 1000.0 / volume.dt) if config.taper_ms_top > 0 else 0
        taper_bottom = int(config.taper_ms_bottom / 1000.0 / volume.dt) if config.taper_ms_bottom > 0 else 0
        if taper_top > 0 or taper_bottom > 0:
            data = apply_temporal_taper(data, taper_top, taper_bottom)

        # Step 2b: Optional temporal pad_copy (for high-amplitude top/bottom edges)
        temporal_pad_top = int(config.pad_time_top_ms / 1000.0 / volume.dt) if config.pad_time_top_ms > 0 else 0
        temporal_pad_bottom = int(config.pad_time_bottom_ms / 1000.0 / volume.dt) if config.pad_time_bottom_ms > 0 else 0
        if temporal_pad_top > 0 or temporal_pad_bottom > 0:
            data = pad_copy_temporal(data, temporal_pad_top, temporal_pad_bottom)
            logger.info(f"  Applied temporal pad_copy: +{temporal_pad_top} samples at top, +{temporal_pad_bottom} at bottom")

        # Step 3: Spatial edge handling with pad_copy
        edge_pad_x = 0
        edge_pad_y = 0

        if config.edge_method == 'pad_copy':
            # Determine padding size (auto or user-specified)
            edge_pad_x = config.pad_traces_x if config.pad_traces_x > 0 else get_auto_pad_size(nx)
            edge_pad_y = config.pad_traces_y if config.pad_traces_y > 0 else get_auto_pad_size(ny)

            # Apply pad_copy: pad with copies of edge traces, taper only padded zone
            data = pad_copy_3d(data, edge_pad_x, edge_pad_y)
            logger.info(f"  Applied pad_copy: +{edge_pad_x} traces per side in X, +{edge_pad_y} traces per side in Y")

        # Step 4: Compute padded dimensions for FFT
        nt_current, nx_current, ny_current = data.shape

        # Apply padding factor for extra padding
        nt_pad = next_power_of_2(int(nt_current * config.padding_factor))
        nx_pad = next_power_of_2(int(nx_current * config.padding_factor))
        ny_pad = next_power_of_2(int(ny_current * config.padding_factor))

        # Ensure at least power-of-2 of current size
        nt_pad = max(nt_pad, next_power_of_2(nt_current))
        nx_pad = max(nx_pad, next_power_of_2(nx_current))
        ny_pad = max(ny_pad, next_power_of_2(ny_current))

        pad_t = nt_pad - nt_current
        pad_x = nx_pad - nx_current
        pad_y = ny_pad - ny_current

        if pad_t > 0 or pad_x > 0 or pad_y > 0:
            data = np.pad(data, ((0, pad_t), (0, pad_x), (0, pad_y)), mode='constant')
            logger.info(f"  FFT padded: ({nt_current},{nx_current},{ny_current}) -> ({nt_pad},{nx_pad},{ny_pad})")

        # Step 5: Compute 3D FFT
        spectrum = self._fft.rfft(data, axis=0)
        spectrum = self._fft.fft(spectrum, axis=1)
        spectrum = self._fft.fft(spectrum, axis=2)
        spectrum = self._fft.fftshift(spectrum, axes=(1, 2))

        # Step 6: Build and apply mask
        mask = self.build_velocity_cone_mask(
            (nt_pad, nx_pad, ny_pad), (volume.dt, volume.dx, volume.dy), config
        )
        spectrum = spectrum * mask

        # Step 7: Inverse 3D FFT
        spectrum = self._fft.ifftshift(spectrum, axes=(1, 2))
        spectrum = self._fft.ifft(spectrum, axis=2)
        spectrum = self._fft.ifft(spectrum, axis=1)
        result = self._fft.irfft(spectrum, n=nt_pad, axis=0)

        # Step 8: Remove FFT padding
        output = result.real.astype(np.float32)[:nt_current, :nx_current, :ny_current]

        # Step 9: Remove temporal pad_copy (extract original time range)
        if temporal_pad_top > 0 or temporal_pad_bottom > 0:
            output = output[temporal_pad_top:temporal_pad_top + nt, :, :]

        # Step 9b: Remove spatial edge padding (extract original data region)
        if config.edge_method == 'pad_copy' and (edge_pad_x > 0 or edge_pad_y > 0):
            output = output[:, edge_pad_x:edge_pad_x + nx, edge_pad_y:edge_pad_y + ny]

        # Step 10: Remove AGC if applied
        if config.apply_agc and agc_scale_factors is not None:
            output = remove_agc_3d(output, agc_scale_factors)

        logger.info("FKK filter applied successfully (CPU)")

        return SeismicVolume(
            data=output, dt=volume.dt, dx=volume.dx, dy=volume.dy,
            metadata={**volume.metadata, 'fkk_filter_applied': True, 'edge_method': config.edge_method}
        )

    def compute_spectrum(self, volume: SeismicVolume) -> np.ndarray:
        """Compute spectrum on CPU."""
        spectrum = self._fft.rfft(volume.data, axis=0)
        spectrum = self._fft.fft(spectrum, axis=1)
        spectrum = self._fft.fft(spectrum, axis=2)
        spectrum = self._fft.fftshift(spectrum, axes=(1, 2))
        self._spectrum_cache = spectrum
        return np.abs(spectrum)

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
        )

        nf, nkx, nky = mask.shape
        mid_f, mid_kx, mid_ky = nf // 2, nkx // 2, nky // 2

        return mask[mid_f, :, :], mask[:, :, mid_ky], mask[:, mid_kx, :]

    def clear_cache(self):
        self._spectrum_cache = None

    def get_status(self) -> Dict[str, Any]:
        """Get processor status."""
        return {
            'device': 'cpu',
            'device_name': 'CPU',
            'gpu_available': False,
        }


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
