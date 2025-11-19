#!/usr/bin/env python3
"""Test actual FK calculation with real SEGY data."""
import segyio
import numpy as np
from scipy import fft
import pandas as pd
from utils.trace_spacing import calculate_trace_spacing_with_stats

segy_path = "/Users/olegadamovich/Downloads/npr3_field.sgy"

print("=" * 80)
print("FK CALCULATION TEST WITH REAL DATA")
print("=" * 80)

# Read first gather
with segyio.open(segy_path, ignore_geometry=True) as f:
    # Assume first 100 traces is one gather
    n_traces_gather = 100

    # Read traces
    traces = np.array([f.trace[i] for i in range(n_traces_gather)])
    traces = traces.T  # Shape: (n_samples, n_traces)

    # Read headers
    headers = []
    for i in range(n_traces_gather):
        h = f.header[i]
        headers.append({
            'offset': h[segyio.TraceField.offset],
            'GroupX': h[segyio.TraceField.GroupX],
            'GroupY': h[segyio.TraceField.GroupY],
            'scalco': h[segyio.TraceField.SourceGroupScalar],
        })

    headers_df = pd.DataFrame(headers)

    # Calculate trace spacing using our function
    stats = calculate_trace_spacing_with_stats(headers_df, default_spacing=25.0)

    print(f"\nGather info:")
    print(f"  Traces shape: {traces.shape}")
    print(f"  Calculated spacing: {stats.spacing:.2f} m (from {stats.coordinate_source})")
    print(f"  Scalar applied: {stats.scalar_applied}")

    # Calculate FK spectrum using same method as FK filter
    n_samples, n_traces = traces.shape
    sample_rate_hz = 250  # Assume 4ms = 250 Hz
    trace_spacing = stats.spacing

    # Forward 2D FFT
    fk_spectrum = fft.fft2(traces)

    # Create frequency and wavenumber axes
    dt = 1.0 / sample_rate_hz
    freqs = fft.fftfreq(n_samples, dt)
    wavenumbers = fft.fftfreq(n_traces, trace_spacing)

    print(f"\n" + "=" * 80)
    print("FK SPECTRUM RESULTS")
    print("=" * 80)

    print(f"\nWavenumber axis:")
    print(f"  Min: {wavenumbers.min():.8f} cycles/m")
    print(f"  Max: {wavenumbers.max():.8f} cycles/m")
    print(f"  Min: {wavenumbers.min() * 1000:.5f} mcycles/m")
    print(f"  Max: {wavenumbers.max() * 1000:.5f} mcycles/m")

    k_nyquist = 1 / (2 * trace_spacing)
    print(f"\n  Nyquist (calculated): {k_nyquist:.8f} cycles/m")
    print(f"  Nyquist (calculated): {k_nyquist * 1000:.5f} mcycles/m")

    print(f"\nFrequency axis:")
    print(f"  Min: {freqs.min():.2f} Hz")
    print(f"  Max: {freqs.max():.2f} Hz")

    # Shift for display
    k_shifted = np.fft.fftshift(wavenumbers)
    freqs_shifted = np.fft.fftshift(freqs)

    print(f"\nShifted wavenumber axis (for plotting):")
    print(f"  Min: {k_shifted.min():.8f} cycles/m = {k_shifted.min() * 1000:.5f} mcycles/m")
    print(f"  Max: {k_shifted.max():.8f} cycles/m = {k_shifted.max() * 1000:.5f} mcycles/m")

    print(f"\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    print(f"\nIf FK plot axis shows values like '-2.5 to 2.5':")
    print(f"  This is likely: ±{k_shifted.max() * 1000:.2f} mcycles/m (millicycles per meter)")
    print(f"  NOT: ±2.5 cycles/m")
    print(f"\nThe 'm' prefix stands for 'milli' (10^-3), not 'meters'")
    print(f"So the plot is CORRECT if it shows ±{k_shifted.max() * 1000:.2f}")
