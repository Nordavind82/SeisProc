"""
Sample seismic data generator for testing.
Creates realistic synthetic seismic data with reflections, noise, and geology.
"""
import numpy as np
import sys
from models.seismic_data import SeismicData


def generate_sample_seismic_data(
    n_samples: int = 1000,
    n_traces: int = 100,
    sample_rate: float = 2.0,
    noise_level: float = 0.1,
    seed: int = 42
) -> SeismicData:
    """
    Generate synthetic seismic data with realistic features.

    Creates data with:
    - Multiple reflection events (horizontal and dipping)
    - Ricker wavelet (typical seismic wavelet)
    - Random noise
    - Lateral velocity variations

    Args:
        n_samples: Number of time samples
        n_traces: Number of traces
        sample_rate: Sample rate in milliseconds
        noise_level: Noise level (0.0 = no noise, 1.0 = high noise)
        seed: Random seed for reproducibility

    Returns:
        SeismicData object with synthetic data
    """
    np.random.seed(seed)

    # Initialize empty traces
    traces = np.zeros((n_samples, n_traces))

    # Time axis in milliseconds
    time_axis = np.arange(n_samples) * sample_rate

    # Generate Ricker wavelet (typical seismic wavelet)
    def ricker_wavelet(t, freq=30.0):
        """Generate Ricker wavelet."""
        t = t - t[len(t)//2]  # Center
        t = t / 1000.0  # Convert ms to seconds
        sigma = 1.0 / (np.pi * freq)
        wavelet = (1.0 - 2.0 * (np.pi * freq * t)**2) * np.exp(-(np.pi * freq * t)**2)
        return wavelet

    wavelet_length = min(100, n_samples // 4)
    wavelet_time = np.arange(wavelet_length) * sample_rate
    wavelet = ricker_wavelet(wavelet_time, freq=30.0)

    # Add multiple reflection events
    reflection_times = [200, 400, 600, 800]  # in ms
    reflection_amplitudes = [1.0, -0.8, 0.6, -0.5]
    reflection_dips = [0.0, 0.3, -0.2, 0.1]  # traces per sample

    for refl_time, amplitude, dip in zip(reflection_times, reflection_amplitudes, reflection_dips):
        if refl_time > time_axis[-1]:
            continue

        for trace_idx in range(n_traces):
            # Calculate time shift due to dip
            time_shift = dip * trace_idx * sample_rate

            # Find sample index for this reflection
            event_time = refl_time + time_shift
            event_sample = int(event_time / sample_rate)

            # Add wavelet at this position
            if 0 <= event_sample < n_samples - wavelet_length:
                traces[event_sample:event_sample+wavelet_length, trace_idx] += \
                    amplitude * wavelet

    # Add lateral velocity variation (creates subtle time shifts)
    for trace_idx in range(n_traces):
        velocity_variation = 0.02 * np.sin(2 * np.pi * trace_idx / n_traces)
        time_stretch = 1.0 + velocity_variation
        stretched_trace = np.interp(
            time_axis * time_stretch,
            time_axis,
            traces[:, trace_idx]
        )
        traces[:, trace_idx] = stretched_trace

    # Add random noise
    if noise_level > 0:
        noise = noise_level * np.random.randn(n_samples, n_traces)
        traces += noise

    # Add some low-frequency background trend
    background = 0.1 * np.sin(2 * np.pi * time_axis / 500.0)[:, np.newaxis]
    traces += background

    # Create SeismicData object
    metadata = {
        'description': 'Synthetic seismic data',
        'generator': 'sample_data.generate_sample_seismic_data',
        'noise_level': noise_level,
        'seed': seed
    }

    return SeismicData(
        traces=traces,
        sample_rate=sample_rate,
        metadata=metadata
    )


def generate_simple_spike_data(
    n_samples: int = 500,
    n_traces: int = 50,
    sample_rate: float = 2.0
) -> SeismicData:
    """
    Generate simple spike test data for debugging.

    Args:
        n_samples: Number of time samples
        n_traces: Number of traces
        sample_rate: Sample rate in milliseconds

    Returns:
        SeismicData with simple spikes
    """
    traces = np.zeros((n_samples, n_traces))

    # Add a few spikes
    spike_times = [100, 250, 400]
    for spike_time in spike_times:
        if spike_time < n_samples:
            traces[spike_time, :] = 1.0

    return SeismicData(
        traces=traces,
        sample_rate=sample_rate,
        metadata={'description': 'Simple spike test data'}
    )
