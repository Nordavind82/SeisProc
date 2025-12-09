"""
Synthetic Pre-Stack Data Generators for Migration Testing

Creates test datasets with known solutions:
- Point diffractor (tests focusing)
- Dipping reflector (tests positioning)
- Flat reflector (tests amplitude preservation)
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass

from models.seismic_data import SeismicData
from models.migration_geometry import MigrationGeometry


@dataclass
class SyntheticEventParams:
    """Parameters for synthetic seismic event."""
    x: float           # Event X position (meters)
    y: float           # Event Y position (meters)
    z: float           # Event depth/time
    amplitude: float   # Reflection amplitude
    wavelet_freq: float  # Dominant frequency (Hz)


def ricker_wavelet(
    t: np.ndarray,
    f_peak: float,
    t0: float = 0.0,
) -> np.ndarray:
    """
    Generate Ricker (Mexican hat) wavelet.

    Args:
        t: Time axis (seconds)
        f_peak: Peak frequency (Hz)
        t0: Time delay (seconds)

    Returns:
        Wavelet amplitude array
    """
    tau = t - t0
    pi_f_tau = np.pi * f_peak * tau
    return (1 - 2 * pi_f_tau**2) * np.exp(-pi_f_tau**2)


def compute_hyperbolic_moveout(
    offset: np.ndarray,
    t0: float,
    velocity: float,
) -> np.ndarray:
    """
    Compute hyperbolic moveout time.

    t(x) = sqrt(t0^2 + (x/v)^2)

    Args:
        offset: Source-receiver offset (meters)
        t0: Zero-offset two-way time (seconds)
        velocity: RMS velocity (m/s)

    Returns:
        Arrival times
    """
    return np.sqrt(t0**2 + (offset / velocity)**2)


def compute_diffractor_traveltime(
    source_x: float,
    source_y: float,
    receiver_x: float,
    receiver_y: float,
    diffractor_x: float,
    diffractor_y: float,
    diffractor_z: float,
    velocity: float,
) -> float:
    """
    Compute two-way traveltime to point diffractor.

    Args:
        source_x, source_y: Source position
        receiver_x, receiver_y: Receiver position
        diffractor_x, diffractor_y, diffractor_z: Diffractor position
        velocity: Velocity (m/s)

    Returns:
        Two-way traveltime (seconds)
    """
    # Source to diffractor
    r_src = np.sqrt(
        (diffractor_x - source_x)**2 +
        (diffractor_y - source_y)**2 +
        diffractor_z**2
    )

    # Diffractor to receiver
    r_rcv = np.sqrt(
        (diffractor_x - receiver_x)**2 +
        (diffractor_y - receiver_y)**2 +
        diffractor_z**2
    )

    return (r_src + r_rcv) / velocity


def create_synthetic_shot_gather(
    n_traces: int = 100,
    n_samples: int = 1000,
    dt_ms: float = 4.0,
    near_offset: float = 100.0,
    far_offset: float = 3000.0,
    events: Optional[List[SyntheticEventParams]] = None,
    velocity: float = 2500.0,
    noise_level: float = 0.05,
    source_x: float = 0.0,
    source_y: float = 0.0,
) -> Tuple[SeismicData, MigrationGeometry]:
    """
    Create synthetic shot gather with hyperbolic events.

    Args:
        n_traces: Number of traces
        n_samples: Samples per trace
        dt_ms: Sample interval (milliseconds)
        near_offset: Minimum offset (meters)
        far_offset: Maximum offset (meters)
        events: List of event parameters (None = default events)
        velocity: RMS velocity (m/s)
        noise_level: Noise amplitude relative to signal
        source_x, source_y: Source position

    Returns:
        Tuple of (SeismicData, MigrationGeometry)
    """
    dt = dt_ms / 1000.0
    t_axis = np.arange(n_samples) * dt

    # Create offset range
    offsets = np.linspace(near_offset, far_offset, n_traces)

    # Initialize traces
    traces = np.zeros((n_samples, n_traces), dtype=np.float32)

    # Default events if none provided
    if events is None:
        events = [
            SyntheticEventParams(x=0, y=0, z=0.5, amplitude=1.0, wavelet_freq=30.0),
            SyntheticEventParams(x=0, y=0, z=1.0, amplitude=0.8, wavelet_freq=25.0),
            SyntheticEventParams(x=0, y=0, z=1.5, amplitude=0.6, wavelet_freq=20.0),
        ]

    # Add events
    for event in events:
        # Compute arrival times for each trace
        t_arrival = compute_hyperbolic_moveout(offsets, event.z, velocity)

        # Add wavelet at arrival time
        for i, t_arr in enumerate(t_arrival):
            if 0 < t_arr < t_axis[-1]:
                wavelet = ricker_wavelet(t_axis, event.wavelet_freq, t_arr)
                traces[:, i] += event.amplitude * wavelet

    # Add noise
    if noise_level > 0:
        rms = np.sqrt(np.mean(traces**2))
        noise = np.random.randn(n_samples, n_traces).astype(np.float32)
        traces += noise_level * rms * noise

    # Create receiver positions (inline from source)
    receiver_x = source_x + offsets
    receiver_y = np.full(n_traces, source_y)

    # Create geometry
    geometry = MigrationGeometry(
        source_x=np.full(n_traces, source_x, dtype=np.float32),
        source_y=np.full(n_traces, source_y, dtype=np.float32),
        receiver_x=receiver_x.astype(np.float32),
        receiver_y=receiver_y.astype(np.float32),
    )

    # Create SeismicData
    data = SeismicData(
        traces=traces,
        sample_rate=dt_ms,
        headers={
            'offset': offsets.astype(np.float32),
        },
        metadata={
            'synthetic': True,
            'velocity': velocity,
            'n_events': len(events),
        }
    )

    return data, geometry


def create_point_diffractor_data(
    diffractor_x: float = 0.0,
    diffractor_y: float = 0.0,
    diffractor_z: float = 1.0,
    velocity: float = 2500.0,
    n_shots: int = 21,
    n_receivers_per_shot: int = 101,
    shot_spacing: float = 100.0,
    receiver_spacing: float = 25.0,
    near_offset: float = 0.0,
    dt_ms: float = 4.0,
    n_samples: int = 1000,
    wavelet_freq: float = 30.0,
    amplitude: float = 1.0,
) -> Tuple[List[SeismicData], List[MigrationGeometry], Dict[str, Any]]:
    """
    Create synthetic data for point diffractor test.

    The diffractor produces a characteristic "smile" pattern when unmigrated.
    After proper migration, it should collapse to a point.

    Args:
        diffractor_x, y, z: Diffractor position
        velocity: Constant velocity (m/s)
        n_shots: Number of shot points
        n_receivers_per_shot: Receivers per shot
        shot_spacing: Distance between shots
        receiver_spacing: Distance between receivers
        near_offset: Minimum offset
        dt_ms: Sample interval (ms)
        n_samples: Samples per trace
        wavelet_freq: Wavelet frequency (Hz)
        amplitude: Event amplitude

    Returns:
        Tuple of (list of gathers, list of geometries, metadata dict)
    """
    dt = dt_ms / 1000.0
    t_axis = np.arange(n_samples) * dt

    # Shot line centered on diffractor X
    shot_start = diffractor_x - (n_shots - 1) * shot_spacing / 2
    shot_positions = shot_start + np.arange(n_shots) * shot_spacing

    gathers = []
    geometries = []

    for shot_idx, shot_x in enumerate(shot_positions):
        shot_y = diffractor_y  # Inline with diffractor

        # Receiver positions (split spread)
        rcv_start = shot_x - (n_receivers_per_shot - 1) * receiver_spacing / 2 + near_offset
        rcv_x = rcv_start + np.arange(n_receivers_per_shot) * receiver_spacing
        rcv_y = np.full(n_receivers_per_shot, shot_y)

        # Initialize traces
        traces = np.zeros((n_samples, n_receivers_per_shot), dtype=np.float32)

        # Compute traveltime and add wavelet for each trace
        for i in range(n_receivers_per_shot):
            t_arr = compute_diffractor_traveltime(
                shot_x, shot_y,
                rcv_x[i], rcv_y[i],
                diffractor_x, diffractor_y, diffractor_z,
                velocity
            )

            if 0 < t_arr < t_axis[-1]:
                wavelet = ricker_wavelet(t_axis, wavelet_freq, t_arr)
                traces[:, i] = amplitude * wavelet

        # Create geometry
        geometry = MigrationGeometry(
            source_x=np.full(n_receivers_per_shot, shot_x, dtype=np.float32),
            source_y=np.full(n_receivers_per_shot, shot_y, dtype=np.float32),
            receiver_x=rcv_x.astype(np.float32),
            receiver_y=rcv_y.astype(np.float32),
        )

        # Create SeismicData
        data = SeismicData(
            traces=traces,
            sample_rate=dt_ms,
            headers={
                'offset': geometry.offset,
                'shot_number': np.full(n_receivers_per_shot, shot_idx, dtype=np.int32),
            },
            metadata={'synthetic': True, 'shot_index': shot_idx}
        )

        gathers.append(data)
        geometries.append(geometry)

    metadata = {
        'test_type': 'point_diffractor',
        'diffractor_position': (diffractor_x, diffractor_y, diffractor_z),
        'velocity': velocity,
        'n_shots': n_shots,
        'n_receivers_per_shot': n_receivers_per_shot,
        'expected_result': 'focused_point',
    }

    return gathers, geometries, metadata


def create_dipping_reflector_data(
    reflector_z0: float = 1.0,
    dip_deg: float = 15.0,
    dip_azimuth: float = 0.0,
    velocity: float = 2500.0,
    n_shots: int = 21,
    n_receivers_per_shot: int = 101,
    shot_spacing: float = 100.0,
    receiver_spacing: float = 25.0,
    dt_ms: float = 4.0,
    n_samples: int = 1000,
    wavelet_freq: float = 30.0,
) -> Tuple[List[SeismicData], List[MigrationGeometry], Dict[str, Any]]:
    """
    Create synthetic data for dipping reflector test.

    The reflector dips in the direction specified by dip_azimuth.

    Args:
        reflector_z0: Depth at x=0, y=0 (seconds for time, meters for depth)
        dip_deg: Reflector dip angle (degrees)
        dip_azimuth: Dip direction (degrees from north/Y)
        velocity: Velocity (m/s)
        n_shots: Number of shots
        n_receivers_per_shot: Receivers per shot
        shot_spacing: Shot interval
        receiver_spacing: Receiver interval
        dt_ms: Sample interval
        n_samples: Samples per trace
        wavelet_freq: Wavelet frequency

    Returns:
        Tuple of (gathers, geometries, metadata)
    """
    dt = dt_ms / 1000.0
    t_axis = np.arange(n_samples) * dt

    # Convert dip to slope
    dip_rad = np.radians(dip_deg)
    az_rad = np.radians(dip_azimuth)

    # Dip components
    dip_x = np.tan(dip_rad) * np.sin(az_rad) / velocity  # time gradient in x
    dip_y = np.tan(dip_rad) * np.cos(az_rad) / velocity  # time gradient in y

    shot_positions = np.arange(n_shots) * shot_spacing

    gathers = []
    geometries = []

    for shot_idx, shot_x in enumerate(shot_positions):
        shot_y = 0.0

        # Receiver positions
        rcv_x = shot_x + np.arange(n_receivers_per_shot) * receiver_spacing
        rcv_y = np.full(n_receivers_per_shot, shot_y)

        traces = np.zeros((n_samples, n_receivers_per_shot), dtype=np.float32)

        for i in range(n_receivers_per_shot):
            # CDP position
            cdp_x = (shot_x + rcv_x[i]) / 2
            cdp_y = (shot_y + rcv_y[i]) / 2

            # Reflector time at CDP
            t0_cdp = reflector_z0 + dip_x * cdp_x + dip_y * cdp_y

            # Offset
            offset = np.sqrt((rcv_x[i] - shot_x)**2 + (rcv_y[i] - shot_y)**2)

            # NMO time (hyperbolic)
            if t0_cdp > 0:
                t_arr = np.sqrt(t0_cdp**2 + (offset / velocity)**2)

                if 0 < t_arr < t_axis[-1]:
                    wavelet = ricker_wavelet(t_axis, wavelet_freq, t_arr)
                    traces[:, i] = wavelet

        geometry = MigrationGeometry(
            source_x=np.full(n_receivers_per_shot, shot_x, dtype=np.float32),
            source_y=np.full(n_receivers_per_shot, shot_y, dtype=np.float32),
            receiver_x=rcv_x.astype(np.float32),
            receiver_y=rcv_y.astype(np.float32),
        )

        data = SeismicData(
            traces=traces,
            sample_rate=dt_ms,
            headers={'offset': geometry.offset},
            metadata={'synthetic': True, 'shot_index': shot_idx}
        )

        gathers.append(data)
        geometries.append(geometry)

    metadata = {
        'test_type': 'dipping_reflector',
        'reflector_z0': reflector_z0,
        'dip_deg': dip_deg,
        'dip_azimuth': dip_azimuth,
        'velocity': velocity,
        'expected_result': 'correctly_positioned_dip',
    }

    return gathers, geometries, metadata


def create_synthetic_3d_survey(
    n_source_lines: int = 5,
    n_sources_per_line: int = 10,
    n_receiver_lines: int = 8,
    n_receivers_per_line: int = 20,
    source_line_spacing: float = 200.0,
    source_spacing: float = 50.0,
    receiver_line_spacing: float = 100.0,
    receiver_spacing: float = 25.0,
    velocity: float = 2500.0,
    reflector_depths: List[float] = None,
    dt_ms: float = 4.0,
    n_samples: int = 1000,
) -> Tuple[List[SeismicData], List[MigrationGeometry], Dict[str, Any]]:
    """
    Create synthetic 3D land survey with orthogonal geometry.

    Args:
        n_source_lines: Number of source lines (Y direction)
        n_sources_per_line: Sources per line (X direction)
        n_receiver_lines: Number of receiver lines (X direction)
        n_receivers_per_line: Receivers per line (Y direction)
        source_line_spacing: Distance between source lines
        source_spacing: Distance between sources on a line
        receiver_line_spacing: Distance between receiver lines
        receiver_spacing: Distance between receivers on a line
        velocity: Background velocity
        reflector_depths: List of reflector depths (seconds)
        dt_ms: Sample interval
        n_samples: Samples per trace

    Returns:
        Tuple of (gathers, geometries, metadata)
    """
    if reflector_depths is None:
        reflector_depths = [0.5, 1.0, 1.5]

    dt = dt_ms / 1000.0
    t_axis = np.arange(n_samples) * dt

    # Generate all source positions
    sources = []
    for line_idx in range(n_source_lines):
        y = line_idx * source_line_spacing
        for src_idx in range(n_sources_per_line):
            x = src_idx * source_spacing
            sources.append((x, y))

    # Generate all receiver positions
    receivers = []
    for line_idx in range(n_receiver_lines):
        x = line_idx * receiver_line_spacing
        for rcv_idx in range(n_receivers_per_line):
            y = rcv_idx * receiver_spacing
            receivers.append((x, y))

    gathers = []
    geometries = []

    for shot_idx, (sx, sy) in enumerate(sources):
        n_receivers = len(receivers)

        traces = np.zeros((n_samples, n_receivers), dtype=np.float32)
        rcv_x = np.array([r[0] for r in receivers], dtype=np.float32)
        rcv_y = np.array([r[1] for r in receivers], dtype=np.float32)

        for i, (rx, ry) in enumerate(receivers):
            offset = np.sqrt((rx - sx)**2 + (ry - sy)**2)

            # Add reflections
            for z0 in reflector_depths:
                t_arr = np.sqrt(z0**2 + (offset / velocity)**2)

                if 0 < t_arr < t_axis[-1]:
                    wavelet = ricker_wavelet(t_axis, 30.0, t_arr)
                    # Amplitude decay with depth
                    amp = 1.0 / (1 + z0)
                    traces[:, i] += amp * wavelet

        geometry = MigrationGeometry(
            source_x=np.full(n_receivers, sx, dtype=np.float32),
            source_y=np.full(n_receivers, sy, dtype=np.float32),
            receiver_x=rcv_x,
            receiver_y=rcv_y,
        )

        data = SeismicData(
            traces=traces,
            sample_rate=dt_ms,
            headers={'offset': geometry.offset},
            metadata={'synthetic': True, 'shot_index': shot_idx}
        )

        gathers.append(data)
        geometries.append(geometry)

    metadata = {
        'test_type': '3d_survey',
        'n_shots': len(sources),
        'n_receivers': len(receivers),
        'velocity': velocity,
        'reflector_depths': reflector_depths,
        'survey_extent_x': (n_sources_per_line - 1) * source_spacing,
        'survey_extent_y': max(
            (n_source_lines - 1) * source_line_spacing,
            (n_receivers_per_line - 1) * receiver_spacing
        ),
    }

    return gathers, geometries, metadata
