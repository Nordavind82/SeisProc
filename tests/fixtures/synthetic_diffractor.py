"""
Synthetic diffractor dataset for PSTM testing.

Creates a zero-offset seismic dataset with a single point diffractor
at a known location. Used for test-driven development of the new
PSTM migration engine.

Diffractor parameters:
- Location: (x=5000m, y=5000m, z=1500m)
- Velocity: 3000 m/s
- Expected traveltime at diffractor location: t = 2*z/v = 1000ms

Grid parameters:
- 100x100 midpoints (10,000 traces)
- Spacing: 50m
- Range: x=[2500, 7450]m, y=[2500, 7450]m
- Time axis: 0-3000ms, dt=2ms (1501 samples)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class DiffractorDataset:
    """Container for synthetic diffractor dataset."""

    # Trace data (n_samples, n_traces)
    traces: np.ndarray

    # Geometry arrays (n_traces,)
    source_x: np.ndarray
    source_y: np.ndarray
    receiver_x: np.ndarray
    receiver_y: np.ndarray
    midpoint_x: np.ndarray
    midpoint_y: np.ndarray

    # Time axis parameters
    dt_ms: float
    t_min_ms: float
    n_samples: int

    # Grid parameters
    n_il: int
    n_xl: int
    il_spacing: float
    xl_spacing: float
    origin_x: float
    origin_y: float

    # Diffractor parameters
    diffractor_x: float
    diffractor_y: float
    diffractor_z: float
    velocity: float

    @property
    def n_traces(self) -> int:
        return self.traces.shape[1]

    @property
    def time_axis_ms(self) -> np.ndarray:
        """Return time axis in milliseconds."""
        return np.arange(self.n_samples) * self.dt_ms + self.t_min_ms

    @property
    def expected_diffractor_time_ms(self) -> float:
        """Expected two-way traveltime at diffractor location."""
        return 2.0 * self.diffractor_z / self.velocity * 1000.0  # ms

    @property
    def expected_il(self) -> int:
        """Expected inline index of diffractor (0-based)."""
        return int((self.diffractor_x - self.origin_x) / self.il_spacing)

    @property
    def expected_xl(self) -> int:
        """Expected crossline index of diffractor (0-based)."""
        return int((self.diffractor_y - self.origin_y) / self.xl_spacing)


def ricker_wavelet(t: np.ndarray, f_peak: float, t_center: float) -> np.ndarray:
    """
    Generate Ricker wavelet (Mexican hat).

    Args:
        t: Time axis in seconds
        f_peak: Peak frequency in Hz
        t_center: Center time of wavelet in seconds

    Returns:
        Wavelet amplitude at each time sample
    """
    tau = t - t_center
    sigma = 1.0 / (np.pi * f_peak * np.sqrt(2))
    amplitude = (1 - (tau / sigma) ** 2) * np.exp(-0.5 * (tau / sigma) ** 2)
    return amplitude


def create_diffractor_dataset(
    # Diffractor parameters
    diffractor_x: float = 5000.0,
    diffractor_y: float = 5000.0,
    diffractor_z: float = 1500.0,
    velocity: float = 3000.0,

    # Grid parameters
    n_il: int = 100,
    n_xl: int = 100,
    il_spacing: float = 50.0,
    xl_spacing: float = 50.0,
    origin_x: float = 2500.0,
    origin_y: float = 2500.0,

    # Time axis
    dt_ms: float = 2.0,
    t_max_ms: float = 3000.0,
    t_min_ms: float = 0.0,

    # Wavelet
    peak_frequency_hz: float = 30.0,
    amplitude: float = 1.0,

    # Options
    add_noise: bool = False,
    noise_level: float = 0.01,
    seed: Optional[int] = 42,
) -> DiffractorDataset:
    """
    Create synthetic zero-offset dataset with single point diffractor.

    Args:
        diffractor_x: X coordinate of diffractor (m)
        diffractor_y: Y coordinate of diffractor (m)
        diffractor_z: Depth of diffractor (m)
        velocity: Constant velocity (m/s)
        n_il: Number of inlines
        n_xl: Number of crosslines
        il_spacing: Inline spacing (m)
        xl_spacing: Crossline spacing (m)
        origin_x: X coordinate of first inline (m)
        origin_y: Y coordinate of first crossline (m)
        dt_ms: Sample interval (ms)
        t_max_ms: Maximum time (ms)
        t_min_ms: Minimum time (ms)
        peak_frequency_hz: Ricker wavelet peak frequency (Hz)
        amplitude: Diffractor amplitude
        add_noise: Add random noise
        noise_level: Noise standard deviation relative to amplitude
        seed: Random seed for reproducibility

    Returns:
        DiffractorDataset containing traces and geometry
    """
    if seed is not None:
        np.random.seed(seed)

    # Create time axis
    n_samples = int((t_max_ms - t_min_ms) / dt_ms) + 1
    time_axis_s = (np.arange(n_samples) * dt_ms + t_min_ms) / 1000.0  # seconds

    # Create midpoint grid
    il_coords = origin_x + np.arange(n_il) * il_spacing
    xl_coords = origin_y + np.arange(n_xl) * xl_spacing

    # Create meshgrid and flatten to trace order (il-major)
    xl_grid, il_grid = np.meshgrid(xl_coords, il_coords)
    midpoint_x = il_grid.flatten().astype(np.float32)
    midpoint_y = xl_grid.flatten().astype(np.float32)

    n_traces = n_il * n_xl

    # For zero-offset: source = receiver = midpoint
    source_x = midpoint_x.copy()
    source_y = midpoint_y.copy()
    receiver_x = midpoint_x.copy()
    receiver_y = midpoint_y.copy()

    # Compute horizontal distance from each midpoint to diffractor
    dx = midpoint_x - diffractor_x
    dy = midpoint_y - diffractor_y
    h = np.sqrt(dx**2 + dy**2)  # (n_traces,)

    # Compute two-way traveltime for each trace (zero-offset)
    # t = 2 * sqrt(h^2 + z^2) / v
    r = np.sqrt(h**2 + diffractor_z**2)
    traveltime_s = 2.0 * r / velocity  # two-way time in seconds

    # Create traces
    traces = np.zeros((n_samples, n_traces), dtype=np.float32)

    # Add Ricker wavelet at computed traveltimes
    for i in range(n_traces):
        t_arrival = traveltime_s[i]
        if t_arrival < time_axis_s[-1]:  # Only add if within time range
            # Geometric spreading amplitude decay: 1/r
            amp = amplitude / (r[i] + 1e-6)
            traces[:, i] = amp * ricker_wavelet(time_axis_s, peak_frequency_hz, t_arrival)

    # Add noise if requested
    if add_noise:
        noise = np.random.randn(n_samples, n_traces).astype(np.float32) * noise_level * amplitude
        traces += noise

    return DiffractorDataset(
        traces=traces,
        source_x=source_x,
        source_y=source_y,
        receiver_x=receiver_x,
        receiver_y=receiver_y,
        midpoint_x=midpoint_x,
        midpoint_y=midpoint_y,
        dt_ms=dt_ms,
        t_min_ms=t_min_ms,
        n_samples=n_samples,
        n_il=n_il,
        n_xl=n_xl,
        il_spacing=il_spacing,
        xl_spacing=xl_spacing,
        origin_x=origin_x,
        origin_y=origin_y,
        diffractor_x=diffractor_x,
        diffractor_y=diffractor_y,
        diffractor_z=diffractor_z,
        velocity=velocity,
    )


def create_expected_migration_result(
    dataset: DiffractorDataset,
    focus_radius_samples: int = 3,
) -> Tuple[np.ndarray, int, int, int]:
    """
    Create analytically expected migration result.

    Perfect migration should collapse the diffractor response
    to a focused point at the diffractor location.

    Args:
        dataset: DiffractorDataset to create expected result for
        focus_radius_samples: Radius of focused energy (accounts for wavelet width)

    Returns:
        expected_image: (n_samples, n_il, n_xl) - expected migrated image
        peak_sample: Expected sample index of peak
        peak_il: Expected inline index of peak
        peak_xl: Expected crossline index of peak
    """
    # Create output image
    expected_image = np.zeros(
        (dataset.n_samples, dataset.n_il, dataset.n_xl),
        dtype=np.float32
    )

    # Expected location of focused diffractor
    peak_il = dataset.expected_il
    peak_xl = dataset.expected_xl

    # Expected time sample (convert ms to sample index)
    peak_time_ms = dataset.expected_diffractor_time_ms
    peak_sample = int((peak_time_ms - dataset.t_min_ms) / dataset.dt_ms)

    # Clamp to valid range
    peak_il = max(0, min(peak_il, dataset.n_il - 1))
    peak_xl = max(0, min(peak_xl, dataset.n_xl - 1))
    peak_sample = max(0, min(peak_sample, dataset.n_samples - 1))

    # Create focused point with Gaussian falloff
    # This models the expected focused energy after migration
    time_axis_s = dataset.time_axis_ms / 1000.0
    t_peak_s = peak_time_ms / 1000.0

    # Temporal wavelet (same as input Ricker, but focused)
    sigma_t = 0.01  # 10ms temporal width
    temporal_profile = np.exp(-0.5 * ((time_axis_s - t_peak_s) / sigma_t) ** 2)

    # Spatial Gaussian (focused point)
    sigma_spatial = 1.5  # ~1.5 grid cells
    for il in range(dataset.n_il):
        for xl in range(dataset.n_xl):
            dist_sq = (il - peak_il)**2 + (xl - peak_xl)**2
            spatial_weight = np.exp(-0.5 * dist_sq / sigma_spatial**2)
            if spatial_weight > 0.01:  # Only compute significant contributions
                expected_image[:, il, xl] = temporal_profile * spatial_weight

    # Normalize so peak = 1
    max_val = expected_image.max()
    if max_val > 0:
        expected_image /= max_val

    return expected_image, peak_sample, peak_il, peak_xl


def verify_dataset(dataset: DiffractorDataset) -> dict:
    """
    Verify synthetic dataset has expected properties.

    Returns dict with verification results.
    """
    results = {}

    # Check shapes
    results['traces_shape'] = dataset.traces.shape
    results['expected_traces_shape'] = (dataset.n_samples, dataset.n_traces)
    results['shapes_match'] = dataset.traces.shape == results['expected_traces_shape']

    # Check geometry
    results['n_traces'] = dataset.n_traces
    results['expected_n_traces'] = dataset.n_il * dataset.n_xl
    results['trace_count_match'] = dataset.n_traces == results['expected_n_traces']

    # Find trace at diffractor location
    dist_to_diff = np.sqrt(
        (dataset.midpoint_x - dataset.diffractor_x)**2 +
        (dataset.midpoint_y - dataset.diffractor_y)**2
    )
    closest_trace_idx = np.argmin(dist_to_diff)
    closest_trace_dist = dist_to_diff[closest_trace_idx]

    results['closest_trace_idx'] = closest_trace_idx
    results['closest_trace_dist_m'] = closest_trace_dist

    # Check that closest trace has event near expected time
    closest_trace = dataset.traces[:, closest_trace_idx]
    peak_sample = np.argmax(np.abs(closest_trace))
    peak_time_ms = peak_sample * dataset.dt_ms + dataset.t_min_ms

    # Expected time for trace at closest location
    r = np.sqrt(closest_trace_dist**2 + dataset.diffractor_z**2)
    expected_time_ms = 2000.0 * r / dataset.velocity

    results['peak_sample'] = peak_sample
    results['peak_time_ms'] = peak_time_ms
    results['expected_time_ms'] = expected_time_ms
    results['time_error_ms'] = abs(peak_time_ms - expected_time_ms)
    results['time_within_tolerance'] = results['time_error_ms'] < dataset.dt_ms * 2

    # Expected time at exact diffractor location
    results['diffractor_expected_time_ms'] = dataset.expected_diffractor_time_ms
    results['diffractor_expected_il'] = dataset.expected_il
    results['diffractor_expected_xl'] = dataset.expected_xl

    return results


# Self-test when run directly
if __name__ == '__main__':
    print("Creating synthetic diffractor dataset...")
    dataset = create_diffractor_dataset()

    print(f"\nDataset created:")
    print(f"  Traces shape: {dataset.traces.shape}")
    print(f"  Number of traces: {dataset.n_traces}")
    print(f"  Grid: {dataset.n_il} x {dataset.n_xl}")
    print(f"  Time axis: {dataset.t_min_ms}-{dataset.time_axis_ms[-1]:.0f}ms, dt={dataset.dt_ms}ms")
    print(f"  Diffractor at: ({dataset.diffractor_x}, {dataset.diffractor_y}, {dataset.diffractor_z})m")
    print(f"  Expected time at diffractor: {dataset.expected_diffractor_time_ms:.1f}ms")
    print(f"  Expected output location: il={dataset.expected_il}, xl={dataset.expected_xl}")

    print("\nVerifying dataset...")
    results = verify_dataset(dataset)
    for key, value in results.items():
        print(f"  {key}: {value}")

    print("\nCreating expected migration result...")
    expected, peak_sample, peak_il, peak_xl = create_expected_migration_result(dataset)
    print(f"  Expected image shape: {expected.shape}")
    print(f"  Peak location: sample={peak_sample}, il={peak_il}, xl={peak_xl}")
    print(f"  Peak value: {expected[peak_sample, peak_il, peak_xl]:.4f}")

    # Verify all checks passed
    all_passed = all([
        results['shapes_match'],
        results['trace_count_match'],
        results['time_within_tolerance'],
    ])

    print(f"\n{'='*50}")
    print(f"All verifications passed: {all_passed}")
    if all_passed:
        print("Dataset generator is working correctly!")
    else:
        print("WARNING: Some verifications failed!")
