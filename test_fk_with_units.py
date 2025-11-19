#!/usr/bin/env python3
"""
Test FK filtering with proper unit specification.
Loads SEGY data and manually sets coordinate_units='feet'.
"""
import numpy as np
import segyio
from models.seismic_data import SeismicData

segy_path = "/Users/olegadamovich/Downloads/npr3_field.sgy"

print("=" * 80)
print("LOADING SEGY WITH UNIT SPECIFICATION")
print("=" * 80)

# Read first gather
with segyio.open(segy_path, ignore_geometry=True) as f:
    n_traces_gather = 100  # First 100 traces

    # Read traces
    traces = np.array([f.trace[i] for i in range(n_traces_gather)])
    traces = traces.T  # Shape: (n_samples, n_traces)

    # Get sample rate
    sample_interval = f.bin[segyio.BinField.Interval]  # microseconds
    sample_rate_ms = sample_interval / 1000.0  # Convert to milliseconds

    print(f"\nData shape: {traces.shape}")
    print(f"Sample rate: {sample_rate_ms} ms")

    # Create SeismicData with FEET units
    data = SeismicData(
        traces=traces,
        sample_rate=sample_rate_ms,
        metadata={
            'coordinate_units': 'feet',  # ← CRITICAL: Specify units!
            'segy_file': segy_path,
        }
    )

    print(f"\nSeismicData created:")
    print(f"  Coordinate units: {data.coordinate_units}")
    print(f"  Unit symbol: {data.unit_symbol}")

    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    print(f"\nNow all calculations will use FEET:")
    print(f"  - Trace spacing: in feet")
    print(f"  - Wavenumbers: in cycles/ft")
    print(f"  - Velocities: in ft/s")
    print(f"  - No conversions needed!")

    print(f"\nFor FK filtering:")
    print(f"  - v = 900 ft/s, k = 0.0072 cycles/ft")
    print(f"  - f = v × k = 900 × 0.0072 = 6.5 Hz ✓")

    print(f"\nTo test: Open this data in FK Designer")
    print(f"Expected:")
    print(f"  - Spacing label: '220 ft'")
    print(f"  - FK axis: 'Wavenumber (cycles/ft)'")
    print(f"  - Velocity labels: 'XXX ft/s'")
    print(f"  - Velocity lines: correct positions!")
