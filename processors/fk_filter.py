"""
FK (Frequency-Wavenumber) Filter Processor

Implements velocity-based filtering in the FK domain for removing
coherent linear noise (ground roll, air wave, etc.) from seismic gathers.

Supports both metric (m/s) and imperial (ft/s) velocity units.
All internal calculations use metric units; conversion happens at boundaries.

Quality improvements (v2.0):
- Spatial windowing to reduce Gibbs ringing artifacts
- Zero-padding option to reduce circular convolution effects
- Consistent cosine taper implementation
- Configurable DC component handling
"""
import numpy as np
from scipy import fft
from scipy.signal.windows import tukey
from typing import Optional, Tuple, Dict
import logging

from processors.base_processor import BaseProcessor
from models.seismic_data import SeismicData
from utils.unit_conversion import (
    convert_velocity_to_metric,
    convert_velocity_from_metric,
    METERS_TO_FEET,
    FEET_TO_METERS
)

logger = logging.getLogger(__name__)


class FKFilter(BaseProcessor):
    """
    FK domain velocity or dip filter.

    Filters seismic gathers based on apparent velocity or dip in the frequency-wavenumber
    domain. Uses 2D FFT to transform from time-space to frequency-wavenumber,
    applies filter, then inverse 2D FFT back to time-space.

    Supports both metric (m/s) and imperial (ft/s) velocity units. Velocities are
    converted to metric internally for calculations, then converted back for display.

    Parameters:
        filter_type: 'velocity' or 'dip' (default: 'velocity')
        v_min: Minimum velocity to pass (in coordinate_units/s)
        v_max: Maximum velocity to pass (in coordinate_units/s)
        v_min_enabled: Enable minimum velocity limit (default: True)
        v_max_enabled: Enable maximum velocity limit (default: True)
        dip_min: Minimum dip to pass (s/coordinate_unit)
        dip_max: Maximum dip to pass (s/coordinate_unit)
        dip_min_enabled: Enable minimum dip limit (default: True)
        dip_max_enabled: Enable maximum dip limit (default: True)
        taper_width: Cosine taper width (velocity_units/s for velocity mode)
        mode: 'pass' or 'reject'
        trace_spacing: Spatial distance between traces (in coordinate_units)
        coordinate_units: 'meters' or 'feet' (default: 'meters')
    """

    def _validate_params(self):
        """Validate FK filter parameters."""
        required = ['taper_width', 'trace_spacing']
        for param in required:
            if param not in self.params:
                raise ValueError(f"Missing required parameter: {param}")

        # Get filter type
        self.filter_type = self.params.get('filter_type', 'velocity')

        # Get coordinate units (meters or feet)
        self.coordinate_units = self.params.get('coordinate_units', 'meters')
        if self.coordinate_units not in ['meters', 'feet']:
            raise ValueError(f"coordinate_units must be 'meters' or 'feet', got {self.coordinate_units}")

        # Common parameters (in display units)
        self.taper_width = float(self.params['taper_width'])
        self.trace_spacing = float(self.params['trace_spacing'])
        self.mode = self.params.get('mode', 'pass')  # 'pass' or 'reject'

        # Quality improvement parameters (v2.0)
        # Spatial windowing: reduces Gibbs ringing from finite aperture
        self.apply_spatial_window = bool(self.params.get('apply_spatial_window', True))
        self.spatial_window_alpha = float(self.params.get('spatial_window_alpha', 0.1))  # Tukey alpha (0-1)

        # Zero-padding: reduces circular convolution artifacts
        self.apply_zero_padding = bool(self.params.get('apply_zero_padding', False))
        self.zero_pad_factor = float(self.params.get('zero_pad_factor', 2.0))  # Pad to N*factor

        # DC component handling
        self.preserve_dc = bool(self.params.get('preserve_dc', True))

        # Velocity mode parameters (in display units - will be converted for internal use)
        self.v_min = float(self.params.get('v_min', 2000.0))
        self.v_max = float(self.params.get('v_max', 6000.0))
        self.v_min_enabled = bool(self.params.get('v_min_enabled', True))
        self.v_max_enabled = bool(self.params.get('v_max_enabled', True))

        # Convert velocities to metric for internal calculations
        # This ensures FK filter math always uses m/s internally
        self._v_min_metric = convert_velocity_to_metric(self.v_min, self.coordinate_units)
        self._v_max_metric = convert_velocity_to_metric(self.v_max, self.coordinate_units)
        self._taper_width_metric = convert_velocity_to_metric(self.taper_width, self.coordinate_units)

        # Convert trace spacing to metric for internal calculations
        if self.coordinate_units == 'feet':
            self._trace_spacing_metric = self.trace_spacing * FEET_TO_METERS
        else:
            self._trace_spacing_metric = self.trace_spacing

        # Dip mode parameters (dip taper is in s/m, convert if in feet)
        self.dip_min = float(self.params.get('dip_min', -0.01))
        self.dip_max = float(self.params.get('dip_max', 0.01))
        self.dip_min_enabled = bool(self.params.get('dip_min_enabled', True))
        self.dip_max_enabled = bool(self.params.get('dip_max_enabled', True))

        # Dip taper width conversion (s/ft to s/m if in feet)
        dip_taper = float(self.params.get('dip_taper_width', self.taper_width / 1000.0))  # Default based on velocity
        if self.coordinate_units == 'feet':
            self._dip_taper_metric = dip_taper * METERS_TO_FEET  # s/ft to s/m
        else:
            self._dip_taper_metric = dip_taper

        # Validation
        if self.taper_width < 0:
            raise ValueError(f"taper_width must be non-negative, got {self.taper_width}")

        if self.trace_spacing <= 0:
            raise ValueError(f"trace_spacing must be positive, got {self.trace_spacing}")

        if self.mode not in ['pass', 'reject']:
            raise ValueError(f"mode must be 'pass' or 'reject', got {self.mode}")

        if self.filter_type not in ['velocity', 'dip']:
            raise ValueError(f"filter_type must be 'velocity' or 'dip', got {self.filter_type}")

        if self.spatial_window_alpha < 0 or self.spatial_window_alpha > 1:
            raise ValueError(f"spatial_window_alpha must be in [0, 1], got {self.spatial_window_alpha}")

        if self.zero_pad_factor < 1:
            raise ValueError(f"zero_pad_factor must be >= 1, got {self.zero_pad_factor}")

        # Velocity mode validation (only if at least one limit is enabled)
        if self.filter_type == 'velocity':
            if self.v_min_enabled and self.v_min <= 0:
                raise ValueError(f"v_min must be positive, got {self.v_min}")

            if self.v_min_enabled and self.v_max_enabled and self.v_max <= self.v_min:
                raise ValueError(
                    f"v_max ({self.v_max}) must be > v_min ({self.v_min})"
                )

    def process(self, data: SeismicData) -> SeismicData:
        """
        Apply FK filter to seismic data.

        Args:
            data: Input seismic data (single gather expected)

        Returns:
            Filtered seismic data
        """
        n_samples, n_traces = data.traces.shape

        # Check minimum trace count
        if n_traces < 8:
            raise ValueError(
                f"FK filtering requires at least 8 traces, got {n_traces}. "
                f"Current gather has too few traces for meaningful FK analysis."
            )

        # Apply FK filter (using metric trace spacing for internal calculations)
        filtered_traces = self._apply_fk_filter(
            data.traces,
            data.sample_rate,
            self._trace_spacing_metric  # Use metric spacing for FFT calculations
        )

        # Create output with metadata
        metadata = data.metadata.copy()
        if 'processing_history' not in metadata:
            metadata['processing_history'] = []
        metadata['processing_history'].append(self.get_description())

        return SeismicData(
            traces=filtered_traces,
            sample_rate=data.sample_rate,
            headers=data.headers,
            metadata=metadata
        )

    def _apply_fk_filter(
        self,
        traces: np.ndarray,
        sample_rate: float,
        trace_spacing: float
    ) -> np.ndarray:
        """
        Apply FK domain filtering with quality improvements.

        Args:
            traces: 2D array (n_samples, n_traces)
            sample_rate: Sample rate in Hz
            trace_spacing: Distance between traces in meters

        Returns:
            Filtered traces (n_samples, n_traces)
        """
        n_samples, n_traces = traces.shape
        original_shape = traces.shape

        # Step 0: Apply spatial windowing (reduces Gibbs ringing)
        if self.apply_spatial_window and self.spatial_window_alpha > 0:
            # Apply Tukey window along spatial dimension (traces)
            spatial_window = tukey(n_traces, alpha=self.spatial_window_alpha)
            traces = traces * spatial_window[np.newaxis, :]
            logger.debug(f"Applied Tukey spatial window (alpha={self.spatial_window_alpha})")

        # Step 0b: Apply zero-padding (reduces circular convolution artifacts)
        if self.apply_zero_padding and self.zero_pad_factor > 1:
            pad_samples = int(n_samples * self.zero_pad_factor) - n_samples
            pad_traces = int(n_traces * self.zero_pad_factor) - n_traces
            traces = np.pad(traces, ((0, pad_samples), (0, pad_traces)), mode='constant')
            logger.debug(f"Zero-padded from {original_shape} to {traces.shape}")

        n_samples_padded, n_traces_padded = traces.shape

        # Step 1: Forward 2D FFT
        fk_spectrum = fft.fft2(traces)

        # Step 2: Create frequency and wavenumber axes
        dt = 1.0 / sample_rate
        freqs = fft.fftfreq(n_samples_padded, dt)  # Temporal frequency (Hz)
        wavenumbers = fft.fftfreq(n_traces_padded, trace_spacing)  # Spatial wavenumber (cycles/m)

        # Create 2D grids
        f_grid, k_grid = np.meshgrid(freqs, wavenumbers, indexing='ij')

        # Step 3: Create filter weights based on filter type
        if self.filter_type == 'velocity':
            weights = self._create_velocity_filter(f_grid, k_grid)
        elif self.filter_type == 'dip':
            weights = self._create_dip_filter(f_grid, k_grid)
        else:
            raise ValueError(f"Unknown filter_type: {self.filter_type}")

        # Step 4: Apply filter in FK domain
        fk_filtered = fk_spectrum * weights

        # Step 5: Inverse 2D FFT
        filtered_traces = fft.ifft2(fk_filtered).real

        # Step 6: Remove zero-padding if applied
        if self.apply_zero_padding and self.zero_pad_factor > 1:
            filtered_traces = filtered_traces[:original_shape[0], :original_shape[1]]

        return filtered_traces

    def _create_velocity_filter(
        self,
        f_grid: np.ndarray,
        k_grid: np.ndarray
    ) -> np.ndarray:
        """
        Create velocity-based filter weights with cosine taper.
        Supports optional min/max velocity limits.

        IMPORTANT: Uses METRIC units internally (m/s) for all calculations.
        Display units are only used for logging.

        Args:
            f_grid: 2D frequency grid (Hz)
            k_grid: 2D wavenumber grid (cycles/m)

        Returns:
            Filter weights (0-1), same shape as input grids
        """
        # Use METRIC units for internal calculations (critical fix!)
        v_min = self._v_min_metric
        v_max = self._v_max_metric
        taper_width = self._taper_width_metric

        logger.debug(f"FK Filter: mode={self.mode}, v_min={v_min:.1f} m/s, "
                     f"v_max={v_max:.1f} m/s, taper={taper_width:.1f} m/s")

        # Calculate apparent velocity at each FK point (in m/s)
        # v_app = f / k
        with np.errstate(divide='ignore', invalid='ignore'):
            v_app = np.abs(f_grid / k_grid)
            # Handle k=0 (infinite velocity)
            v_app[k_grid == 0] = np.inf

        # Initialize weights based on mode
        if self.mode == 'pass':
            weights = np.ones_like(v_app)
        else:  # 'reject'
            weights = np.zeros_like(v_app)

        # Define taper zones using METRIC units
        if self.v_min_enabled:
            v1 = max(0, v_min - taper_width)  # Lower bound of taper (ensure >= 0)
            v2 = v_min + taper_width           # Upper bound of taper
        if self.v_max_enabled:
            v3 = v_max - taper_width           # Lower bound of upper taper
            v4 = v_max + taper_width           # Upper bound of upper taper

        if self.mode == 'pass':
            # Pass band: keep velocities between v_min and v_max (if enabled)

            # Apply minimum velocity limit (if enabled)
            if self.v_min_enabled:
                # Full reject: v < v1
                weights[v_app < v1] = 0.0

                # Taper up: v1 <= v < v2 (use consistent raised cosine)
                mask = (v_app >= v1) & (v_app < v2)
                if taper_width > 0 and np.any(mask):
                    # Normalized position in taper zone [0, 1]
                    t = (v_app[mask] - v1) / (2.0 * taper_width)
                    weights[mask] = 0.5 * (1.0 - np.cos(np.pi * t))

            # Apply maximum velocity limit (if enabled)
            if self.v_max_enabled:
                # Full reject: v > v4
                weights[v_app > v4] = 0.0

                # Taper down: v3 < v <= v4 (consistent with min taper)
                mask = (v_app > v3) & (v_app <= v4)
                if taper_width > 0 and np.any(mask):
                    # Normalized position in taper zone [0, 1], reversed for taper-down
                    t = (v_app[mask] - v3) / (2.0 * taper_width)
                    weights[mask] = 0.5 * (1.0 + np.cos(np.pi * t))

        else:  # mode == 'reject'
            # Reject band: remove velocities between v_min and v_max (if enabled)

            if not self.v_min_enabled and not self.v_max_enabled:
                weights = np.ones_like(v_app)
            else:
                weights = np.ones_like(v_app)

                if self.v_min_enabled and self.v_max_enabled:
                    # Standard reject band between v_min and v_max

                    # Taper down entering rejection zone: v1 <= v < v2
                    mask = (v_app >= v1) & (v_app < v2)
                    if taper_width > 0 and np.any(mask):
                        t = (v_app[mask] - v1) / (2.0 * taper_width)
                        weights[mask] = 0.5 * (1.0 + np.cos(np.pi * t))
                    elif np.any(mask):
                        weights[mask] = 0.0

                    # Full reject: v2 <= v <= v3
                    weights[(v_app >= v2) & (v_app <= v3)] = 0.0

                    # Taper up exiting rejection zone: v3 < v <= v4
                    mask = (v_app > v3) & (v_app <= v4)
                    if taper_width > 0 and np.any(mask):
                        t = (v_app[mask] - v3) / (2.0 * taper_width)
                        weights[mask] = 0.5 * (1.0 - np.cos(np.pi * t))
                    elif np.any(mask):
                        weights[mask] = 0.0

                elif self.v_min_enabled:
                    # Only reject v < v_min (low velocities)

                    # Taper: v1 <= v < v2
                    mask = (v_app >= v1) & (v_app < v2)
                    if taper_width > 0 and np.any(mask):
                        t = (v_app[mask] - v1) / (2.0 * taper_width)
                        weights[mask] = 0.5 * (1.0 - np.cos(np.pi * t))

                    # Full reject: v < v1
                    weights[v_app < v1] = 0.0

                elif self.v_max_enabled:
                    # Only reject v > v_max (high velocities)

                    # Taper: v3 < v <= v4
                    mask = (v_app > v3) & (v_app <= v4)
                    if taper_width > 0 and np.any(mask):
                        t = (v_app[mask] - v3) / (2.0 * taper_width)
                        weights[mask] = 0.5 * (1.0 - np.cos(np.pi * t))

                    # Full reject: v > v4
                    weights[v_app > v4] = 0.0

        # Handle DC component (f=0, k=0) - configurable
        if self.preserve_dc:
            weights[0, 0] = 1.0
        # If preserve_dc=False in reject mode, DC can be rejected

        # Log filter statistics
        n_zero = np.sum(weights == 0.0)
        n_one = np.sum(weights == 1.0)
        n_taper = np.sum((weights > 0.0) & (weights < 1.0))
        logger.debug(f"Filter weights: reject={100*n_zero/weights.size:.1f}%, "
                     f"pass={100*n_one/weights.size:.1f}%, taper={100*n_taper/weights.size:.1f}%")

        return weights

    def _create_dip_filter(
        self,
        f_grid: np.ndarray,
        k_grid: np.ndarray
    ) -> np.ndarray:
        """
        Create dip-based filter weights with cosine taper.
        Supports optional min/max dip limits.

        Dip is defined as dt/dx (s/m). In FK domain: dip = k/f
        - Negative dip: k < 0 (left-dipping events)
        - Positive dip: k > 0 (right-dipping events)

        IMPORTANT: Uses METRIC units internally (s/m) for all calculations.

        Args:
            f_grid: 2D frequency grid (Hz)
            k_grid: 2D wavenumber grid (cycles/m)

        Returns:
            Filter weights (0-1), same shape as input grids
        """
        # Use metric taper width for dip mode
        taper_width = self._dip_taper_metric

        logger.debug(f"Dip Filter: mode={self.mode}, dip_min={self.dip_min:.4f} s/m, "
                     f"dip_max={self.dip_max:.4f} s/m, taper={taper_width:.4f} s/m")

        # Calculate apparent dip at each FK point
        # dip = k / f (units: (cycles/m) / Hz = s/m)
        with np.errstate(divide='ignore', invalid='ignore'):
            dip_app = k_grid / f_grid
            # Handle f=0 (infinite dip)
            dip_app[f_grid == 0] = np.inf

        # Initialize weights based on mode
        if self.mode == 'pass':
            weights = np.ones_like(dip_app)
        else:  # 'reject'
            weights = np.zeros_like(dip_app)

        # Define taper zones for enabled limits
        # Note: dip_min is typically negative (left-dipping), dip_max is positive (right-dipping)
        if self.dip_min_enabled:
            d1 = self.dip_min - taper_width  # More negative
            d2 = self.dip_min + taper_width  # Less negative (toward zero)
        if self.dip_max_enabled:
            d3 = self.dip_max - taper_width  # Less positive (toward zero)
            d4 = self.dip_max + taper_width  # More positive

        if self.mode == 'pass':
            # Pass band: keep dips between dip_min and dip_max (if enabled)

            if self.dip_min_enabled:
                # Full reject: dip < d1
                weights[dip_app < d1] = 0.0

                # Taper up: d1 <= dip < d2
                mask = (dip_app >= d1) & (dip_app < d2)
                if taper_width > 0 and np.any(mask):
                    t = (dip_app[mask] - d1) / (2.0 * taper_width)
                    weights[mask] = 0.5 * (1.0 - np.cos(np.pi * t))

            if self.dip_max_enabled:
                # Full reject: dip > d4
                weights[dip_app > d4] = 0.0

                # Taper down: d3 < dip <= d4
                mask = (dip_app > d3) & (dip_app <= d4)
                if taper_width > 0 and np.any(mask):
                    t = (dip_app[mask] - d3) / (2.0 * taper_width)
                    weights[mask] = 0.5 * (1.0 + np.cos(np.pi * t))

        else:  # mode == 'reject'
            # Reject band: remove dips between dip_min and dip_max

            if not self.dip_min_enabled and not self.dip_max_enabled:
                weights = np.ones_like(dip_app)
            else:
                weights = np.ones_like(dip_app)

                if self.dip_min_enabled and self.dip_max_enabled:
                    # Taper down entering rejection zone: d1 <= dip < d2
                    mask = (dip_app >= d1) & (dip_app < d2)
                    if taper_width > 0 and np.any(mask):
                        t = (dip_app[mask] - d1) / (2.0 * taper_width)
                        weights[mask] = 0.5 * (1.0 + np.cos(np.pi * t))
                    elif np.any(mask):
                        weights[mask] = 0.0

                    # Full reject: d2 <= dip <= d3
                    weights[(dip_app >= d2) & (dip_app <= d3)] = 0.0

                    # Taper up exiting rejection zone: d3 < dip <= d4
                    mask = (dip_app > d3) & (dip_app <= d4)
                    if taper_width > 0 and np.any(mask):
                        t = (dip_app[mask] - d3) / (2.0 * taper_width)
                        weights[mask] = 0.5 * (1.0 - np.cos(np.pi * t))
                    elif np.any(mask):
                        weights[mask] = 0.0

                elif self.dip_min_enabled:
                    # Taper: d1 <= dip < d2
                    mask = (dip_app >= d1) & (dip_app < d2)
                    if taper_width > 0 and np.any(mask):
                        t = (dip_app[mask] - d1) / (2.0 * taper_width)
                        weights[mask] = 0.5 * (1.0 - np.cos(np.pi * t))

                    # Full reject: dip < d1
                    weights[dip_app < d1] = 0.0

                elif self.dip_max_enabled:
                    # Taper: d3 < dip <= d4
                    mask = (dip_app > d3) & (dip_app <= d4)
                    if taper_width > 0 and np.any(mask):
                        t = (dip_app[mask] - d3) / (2.0 * taper_width)
                        weights[mask] = 0.5 * (1.0 - np.cos(np.pi * t))

                    # Full reject: dip > d4
                    weights[dip_app > d4] = 0.0

        # Handle DC component - configurable
        if self.preserve_dc:
            weights[0, 0] = 1.0

        # Log filter statistics
        n_zero = np.sum(weights == 0.0)
        n_one = np.sum(weights == 1.0)
        n_taper = np.sum((weights > 0.0) & (weights < 1.0))
        logger.debug(f"Dip filter weights: reject={100*n_zero/weights.size:.1f}%, "
                     f"pass={100*n_one/weights.size:.1f}%, taper={100*n_taper/weights.size:.1f}%")

        return weights

    def compute_fk_spectrum(
        self,
        traces: np.ndarray,
        sample_rate: float,
        trace_spacing: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute FK spectrum for visualization.

        Args:
            traces: 2D array (n_samples, n_traces)
            sample_rate: Sample rate in Hz
            trace_spacing: Distance between traces in meters

        Returns:
            Tuple of (spectrum, freqs, wavenumbers)
            - spectrum: 2D complex FK spectrum
            - freqs: 1D frequency axis (Hz)
            - wavenumbers: 1D wavenumber axis (cycles/m)
        """
        n_samples, n_traces = traces.shape

        # Forward 2D FFT
        fk_spectrum = fft.fft2(traces)

        # Frequency and wavenumber axes
        dt = 1.0 / sample_rate
        freqs = fft.fftfreq(n_samples, dt)
        wavenumbers = fft.fftfreq(n_traces, trace_spacing)

        return fk_spectrum, freqs, wavenumbers

    def get_description(self) -> str:
        """Get description of this filter with correct units."""
        mode_str = "Pass" if self.mode == 'pass' else "Reject"
        unit = 'ft' if self.coordinate_units == 'feet' else 'm'
        return (
            f"FK Filter ({mode_str}): {self.v_min:.0f}-{self.v_max:.0f} {unit}/s, "
            f"taper {self.taper_width:.0f} {unit}/s"
        )


# Presets stored in metric (m/s) - converted to display units when used
_FK_PRESETS_METRIC = {
    'Ground Roll Removal': {
        'v_min': 1500,
        'v_max': 6000,
        'taper_width': 300,
        'mode': 'pass',
        'description': 'Removes slow horizontal noise (ground roll)'
    },
    'Air Wave Removal': {
        'v_min': 400,
        'v_max': 10000,
        'taper_width': 100,
        'mode': 'pass',
        'description': 'Removes direct air wave (<400 m/s)'
    },
    'Reflection Pass': {
        'v_min': 2000,
        'v_max': 5000,
        'taper_width': 200,
        'mode': 'pass',
        'description': 'Keeps typical reflection velocities'
    },
    'Steep Dip Only': {
        'v_min': 4000,
        'v_max': 8000,
        'taper_width': 400,
        'mode': 'pass',
        'description': 'Keeps only steep events'
    }
}


def get_fk_filter_presets(coordinate_units: str = 'meters') -> Dict[str, Dict]:
    """
    Get predefined FK filter configurations.

    Args:
        coordinate_units: Target units for velocities ('meters' or 'feet')

    Returns:
        Dictionary of preset name -> parameters (in specified units)
    """
    presets = {}
    for name, params in _FK_PRESETS_METRIC.items():
        preset = params.copy()
        # Convert velocities if feet mode requested
        if coordinate_units == 'feet':
            preset['v_min'] = convert_velocity_from_metric(params['v_min'], 'feet')
            preset['v_max'] = convert_velocity_from_metric(params['v_max'], 'feet')
            preset['taper_width'] = convert_velocity_from_metric(params['taper_width'], 'feet')
            # Update description to show ft/s
            preset['description'] = preset['description'].replace('m/s', 'ft/s')
        preset['coordinate_units'] = coordinate_units
        presets[name] = preset
    return presets
