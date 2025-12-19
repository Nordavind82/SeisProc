"""
Synthetic Seismic Data Generator for Benchmarking

Generates realistic crosspread gathers with:
- Configurable geometry (50m source/receiver spacing default)
- Multiple reflections with realistic frequency content
- Additive noise
- Proper headers for FKK and other processing

Output format matches SeisProc's Zarr/Parquet storage.
"""

import numpy as np
import pandas as pd
import zarr
import json
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass
import time


@dataclass
class CrossspreadGeometry:
    """Crosspread acquisition geometry parameters."""
    n_gathers: int = 100
    traces_per_gather: int = 4000
    n_samples: int = 1600
    sample_rate_ms: float = 2.0

    # Spatial parameters
    source_spacing: float = 50.0      # meters
    receiver_spacing: float = 50.0    # meters
    inline_spacing: float = 25.0      # meters (CDP bin size)
    xline_spacing: float = 12.5       # meters (CDP bin size)

    # Spread geometry
    n_receivers_per_line: int = 80    # receivers per receiver line
    n_receiver_lines: int = 50        # number of receiver lines

    @property
    def total_traces(self) -> int:
        return self.n_gathers * self.traces_per_gather

    @property
    def duration_ms(self) -> float:
        return self.n_samples * self.sample_rate_ms

    @property
    def nyquist_freq(self) -> float:
        return 500.0 / self.sample_rate_ms  # Hz


def ricker_wavelet(t: np.ndarray, f0: float, t0: float) -> np.ndarray:
    """
    Generate Ricker wavelet.

    Args:
        t: Time array in seconds
        f0: Central frequency in Hz
        t0: Peak time in seconds

    Returns:
        Wavelet amplitude array
    """
    tau = t - t0
    pi_f_tau = np.pi * f0 * tau
    return (1 - 2 * pi_f_tau**2) * np.exp(-pi_f_tau**2)


def generate_reflection_times(
    offset: float,
    velocities: np.ndarray,
    t0_times: np.ndarray
) -> np.ndarray:
    """
    Calculate NMO-corrected reflection times for given offset.

    Args:
        offset: Source-receiver offset in meters
        velocities: RMS velocities at each reflector (m/s)
        t0_times: Zero-offset times in seconds

    Returns:
        Reflection times at given offset
    """
    # Hyperbolic moveout: t = sqrt(t0^2 + (x/v)^2)
    return np.sqrt(t0_times**2 + (offset / velocities)**2)


def generate_synthetic_gather(
    geometry: CrossspreadGeometry,
    gather_idx: int,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Generate a single synthetic crosspread gather.

    Args:
        geometry: Geometry parameters
        gather_idx: Gather index (affects source position)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (traces array [n_samples, n_traces], headers DataFrame)
    """
    if seed is not None:
        np.random.seed(seed + gather_idx)

    n_traces = geometry.traces_per_gather
    n_samples = geometry.n_samples
    dt = geometry.sample_rate_ms / 1000.0  # Convert to seconds

    # Time axis
    t = np.arange(n_samples) * dt

    # Source position for this gather (moves along inline direction)
    source_x = gather_idx * geometry.source_spacing
    source_y = 0.0

    # Generate receiver positions (crosspread pattern)
    # Receivers on lines perpendicular to shot line
    receiver_x = []
    receiver_y = []

    n_receivers_inline = int(np.ceil(np.sqrt(n_traces)))
    n_receivers_xline = int(np.ceil(n_traces / n_receivers_inline))

    for ix in range(n_receivers_inline):
        for iy in range(n_receivers_xline):
            if len(receiver_x) >= n_traces:
                break
            rx = source_x + (ix - n_receivers_inline // 2) * geometry.receiver_spacing
            ry = (iy - n_receivers_xline // 2) * geometry.receiver_spacing
            receiver_x.append(rx)
            receiver_y.append(ry)
        if len(receiver_x) >= n_traces:
            break

    # Ensure exact trace count
    receiver_x = np.array(receiver_x[:n_traces], dtype=np.float32)
    receiver_y = np.array(receiver_y[:n_traces], dtype=np.float32)

    # Calculate offsets and azimuths
    dx = receiver_x - source_x
    dy = receiver_y - source_y
    offsets = np.sqrt(dx**2 + dy**2)
    azimuths = np.degrees(np.arctan2(dy, dx)) % 360

    # CDP positions (midpoint)
    cdp_x = (source_x + receiver_x) / 2.0
    cdp_y = (source_y + receiver_y) / 2.0

    # Inline/crossline from CDP
    inline = np.round(cdp_x / geometry.inline_spacing).astype(int)
    crossline = np.round(cdp_y / geometry.xline_spacing).astype(int)

    # Define reflectors
    n_reflectors = 8
    t0_times = np.array([0.3, 0.5, 0.8, 1.1, 1.5, 1.9, 2.3, 2.8])  # seconds
    velocities = np.array([1800, 2000, 2200, 2500, 2800, 3100, 3400, 3800])  # m/s
    amplitudes = np.array([1.0, 0.8, 1.2, 0.6, 0.9, 0.7, 0.5, 0.4])
    frequencies = np.array([35, 30, 25, 22, 20, 18, 16, 14])  # Hz

    # Generate traces
    traces = np.zeros((n_samples, n_traces), dtype=np.float32)

    for i in range(n_traces):
        offset = offsets[i]

        # Calculate reflection times for this offset
        refl_times = generate_reflection_times(offset, velocities, t0_times)

        # Generate reflections
        for j in range(n_reflectors):
            if refl_times[j] < t[-1]:  # Only if within record length
                # Wavelet with some amplitude variation
                amp_var = 1.0 + 0.1 * np.random.randn()
                wavelet = ricker_wavelet(t, frequencies[j], refl_times[j])
                traces[:, i] += amplitudes[j] * amp_var * wavelet

        # Add geometric spreading (amplitude decay with offset)
        spreading = 1.0 / (1.0 + 0.0005 * offset)
        traces[:, i] *= spreading

    # Add coherent noise (ground roll - low velocity)
    ground_roll_velocity = 400  # m/s (very slow)
    for i in range(n_traces):
        offset = offsets[i]
        t_gr = offset / ground_roll_velocity
        if t_gr < t[-1]:
            # Low frequency wavelet for ground roll
            gr_wavelet = ricker_wavelet(t, 8, t_gr) * 0.3
            traces[:, i] += gr_wavelet

    # Add random noise
    noise_level = 0.05 * np.std(traces)
    traces += np.random.randn(n_samples, n_traces).astype(np.float32) * noise_level

    # Build headers DataFrame
    headers = pd.DataFrame({
        'trace_index': np.arange(n_traces),
        'field_record': np.full(n_traces, gather_idx),
        'trace_in_field': np.arange(n_traces),
        'source_x': np.full(n_traces, source_x),
        'source_y': np.full(n_traces, source_y),
        'receiver_x': receiver_x,
        'receiver_y': receiver_y,
        'cdp_x': cdp_x,
        'cdp_y': cdp_y,
        'offset': offsets,
        'azimuth': azimuths,
        'inline': inline,
        'crossline': crossline,
    })

    return traces, headers


def generate_benchmark_dataset(
    output_dir: str,
    geometry: Optional[CrossspreadGeometry] = None,
    show_progress: bool = True
) -> Path:
    """
    Generate complete benchmark dataset and save to Zarr/Parquet.

    Args:
        output_dir: Output directory path
        geometry: Geometry parameters (uses defaults if None)
        show_progress: Print progress messages

    Returns:
        Path to output directory
    """
    if geometry is None:
        geometry = CrossspreadGeometry()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    traces_path = output_path / 'traces.zarr'
    headers_path = output_path / 'headers.parquet'
    ensemble_path = output_path / 'ensemble_index.parquet'
    metadata_path = output_path / 'metadata.json'

    if show_progress:
        print(f"Generating synthetic benchmark dataset:")
        print(f"  Gathers: {geometry.n_gathers}")
        print(f"  Traces per gather: {geometry.traces_per_gather}")
        print(f"  Total traces: {geometry.total_traces:,}")
        print(f"  Samples: {geometry.n_samples} ({geometry.duration_ms:.0f} ms @ {geometry.sample_rate_ms} ms)")
        print(f"  Output: {output_path}")

    start_time = time.time()

    # Create Zarr array for traces (n_samples, n_traces) - transposed format
    z = zarr.open(
        str(traces_path),
        mode='w',
        shape=(geometry.n_samples, geometry.total_traces),
        chunks=(geometry.n_samples, min(1000, geometry.traces_per_gather)),
        dtype=np.float32,
        compressor=None,  # No compression for speed
        zarr_format=2
    )

    # Storage for headers and ensemble index
    all_headers = []
    ensemble_data = []

    global_trace_idx = 0

    for gather_idx in range(geometry.n_gathers):
        # Generate gather
        traces, headers = generate_synthetic_gather(geometry, gather_idx, seed=42)

        # Update global trace indices
        n_traces = traces.shape[1]
        headers['trace_index'] = np.arange(global_trace_idx, global_trace_idx + n_traces)

        # Write traces to Zarr
        z[:, global_trace_idx:global_trace_idx + n_traces] = traces

        # Store headers
        all_headers.append(headers)

        # Store ensemble boundary
        ensemble_data.append({
            'ensemble_id': gather_idx,
            'field_record': gather_idx,
            'start_trace': global_trace_idx,
            'end_trace': global_trace_idx + n_traces - 1,
            'n_traces': n_traces
        })

        global_trace_idx += n_traces

        if show_progress and (gather_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (gather_idx + 1) / elapsed
            remaining = (geometry.n_gathers - gather_idx - 1) / rate
            print(f"  Generated {gather_idx + 1}/{geometry.n_gathers} gathers "
                  f"({elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining)")

    # Save headers to Parquet
    df_headers = pd.concat(all_headers, ignore_index=True)
    df_headers.to_parquet(headers_path, engine='pyarrow', compression='snappy', index=False)

    # Save ensemble index
    df_ensembles = pd.DataFrame(ensemble_data)
    df_ensembles.to_parquet(ensemble_path, engine='pyarrow', compression='snappy', index=False)

    # Save metadata
    metadata = {
        'shape': [geometry.n_samples, geometry.total_traces],
        'n_samples': geometry.n_samples,
        'n_traces': geometry.total_traces,
        'sample_rate': geometry.sample_rate_ms,
        'duration_ms': geometry.duration_ms,
        'nyquist_freq': geometry.nyquist_freq,
        'n_gathers': geometry.n_gathers,
        'traces_per_gather': geometry.traces_per_gather,
        'seismic_metadata': {
            'data_type': 'synthetic_benchmark',
            'geometry': 'crosspread',
            'source_spacing_m': geometry.source_spacing,
            'receiver_spacing_m': geometry.receiver_spacing,
        },
        'storage_info': {
            'zarr_chunks': f"({geometry.n_samples}, 1000)",
            'parquet_compression': 'snappy',
            'zarr_compression': 'none'
        }
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - start_time

    # Calculate sizes
    zarr_size = sum(f.stat().st_size for f in traces_path.rglob('*') if f.is_file())
    headers_size = headers_path.stat().st_size
    total_size = zarr_size + headers_size

    if show_progress:
        print(f"\nDataset generated successfully!")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Zarr size: {zarr_size / 1024 / 1024:.1f} MB")
        print(f"  Headers size: {headers_size / 1024 / 1024:.1f} MB")
        print(f"  Total size: {total_size / 1024 / 1024:.1f} MB")
        print(f"  Throughput: {geometry.total_traces / elapsed:.0f} traces/s")

    return output_path


def load_benchmark_gather(
    data_dir: str,
    gather_idx: int
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load a single gather from benchmark dataset.

    Args:
        data_dir: Path to benchmark dataset
        gather_idx: Gather index to load

    Returns:
        Tuple of (traces [n_samples, n_traces], headers DataFrame)
    """
    data_path = Path(data_dir)

    # Load ensemble index to get trace range
    df_ensembles = pd.read_parquet(data_path / 'ensemble_index.parquet')
    ensemble = df_ensembles[df_ensembles['ensemble_id'] == gather_idx].iloc[0]

    start = int(ensemble['start_trace'])
    end = int(ensemble['end_trace'])

    # Load traces from Zarr
    z = zarr.open(str(data_path / 'traces.zarr'), mode='r')
    traces = z[:, start:end+1]

    # Load headers
    df_headers = pd.read_parquet(data_path / 'headers.parquet')
    headers = df_headers[(df_headers['trace_index'] >= start) &
                         (df_headers['trace_index'] <= end)].copy()

    return traces, headers


def load_benchmark_metadata(data_dir: str) -> dict:
    """Load metadata from benchmark dataset."""
    with open(Path(data_dir) / 'metadata.json', 'r') as f:
        return json.load(f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate synthetic benchmark dataset')
    parser.add_argument('--output', '-o', type=str, default='benchmark_data',
                        help='Output directory')
    parser.add_argument('--gathers', '-g', type=int, default=100,
                        help='Number of gathers')
    parser.add_argument('--traces', '-t', type=int, default=4000,
                        help='Traces per gather')
    parser.add_argument('--samples', '-s', type=int, default=1600,
                        help='Samples per trace')
    parser.add_argument('--sample-rate', '-r', type=float, default=2.0,
                        help='Sample rate in ms')

    args = parser.parse_args()

    geometry = CrossspreadGeometry(
        n_gathers=args.gathers,
        traces_per_gather=args.traces,
        n_samples=args.samples,
        sample_rate_ms=args.sample_rate
    )

    generate_benchmark_dataset(args.output, geometry)
