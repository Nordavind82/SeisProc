#!/usr/bin/env python3
"""
Test to understand why median spacing comes out wrong.
"""
import numpy as np
import pandas as pd
from utils.trace_spacing import calculate_trace_spacing_with_stats

print("=" * 70)
print("TESTING DIFFERENT RECEIVER CONFIGURATIONS")
print("=" * 70)

# Test different configurations
configurations = [
    # (receivers_per_shot, receiver_spacing, shot_spacing)
    (45, 33, 220),  # Dense receivers
    (5, 33, 220),   # Sparse receivers
    (3, 33, 220),   # Very sparse receivers
    (2, 33, 220),   # Minimal receivers (2 per shot)
]

for n_receivers, rx_spacing, shot_spacing in configurations:
    print(f"\n{'='*70}")
    print(f"Configuration: {n_receivers} receivers/shot, {rx_spacing}m spacing, {shot_spacing}m shots")
    print('='*70)

    n_shots = 11
    coords_list = []

    for shot in range(n_shots):
        shot_start = shot * shot_spacing
        for rx in range(n_receivers):
            coords_list.append(shot_start + rx * rx_spacing)

    coords = np.array(coords_list)
    total_traces = len(coords)

    print(f"Total traces: {total_traces}")

    # Calculate spacings
    spacings = np.abs(np.diff(coords))
    spacings_nonzero = spacings[spacings > 0]

    # Count spacing types
    small_spacings = spacings_nonzero[spacings_nonzero < 100]  # Receiver spacings
    large_spacings = spacings_nonzero[spacings_nonzero >= 100]  # Shot gaps

    median_spacing = np.median(spacings_nonzero)

    print(f"\nSpacing breakdown:")
    print(f"  Receiver spacings (< 100m): {len(small_spacings)} occurrences, median = {np.median(small_spacings) if len(small_spacings) > 0 else 0:.1f} m")
    print(f"  Shot gaps (>= 100m): {len(large_spacings)} occurrences, median = {np.median(large_spacings) if len(large_spacings) > 0 else 0:.1f} m")
    print(f"\n  Overall median spacing: {median_spacing:.1f} m")

    # Calculate what FK range this would give
    k_max = 1000 / (2 * median_spacing)  # mcycles/m
    k_correct = 1000 / (2 * rx_spacing)  # mcycles/m

    print(f"\n  FK wavenumber range with this spacing: ±{k_max:.3f} mcycles/m")
    print(f"  FK wavenumber range with correct spacing: ±{k_correct:.3f} mcycles/m")

    if median_spacing > rx_spacing * 1.5:
        print(f"\n  ⚠️  MEDIAN IS DOMINATED BY GAPS!")
        print(f"  Calculated spacing ({median_spacing:.1f}m) is {median_spacing/rx_spacing:.1f}x too large")

# Now let's figure out what configuration matches the user's data
print(f"\n{'='*70}")
print("REVERSE ENGINEERING USER'S DATA")
print('='*70)

observed_median = 219.9  # m (from screenshot)
observed_k_max = 15.0    # mcycles/m (from screenshot)
n_traces = 497
n_shots_estimated = 11  # from screenshot "Current: 1/11"

# From FK range, calculate actual spacing
actual_spacing = 1000 / (2 * observed_k_max)
print(f"\nFrom FK plot (±{observed_k_max} mcycles/m):")
print(f"  Actual receiver spacing: {actual_spacing:.1f} m")

# If median is 219.9m but actual is 33.3m, the median must be dominated by gaps
print(f"\nFrom reported median ({observed_median}m):")
print(f"  This suggests most trace-to-trace spacings are large gaps, not receiver spacings")

receivers_per_shot = n_traces / n_shots_estimated
print(f"\nEstimated: {receivers_per_shot:.1f} traces per shot")

# If there are only ~2 receivers per shot:
# - Each shot contributes 1 receiver spacing (33m)
# - Each shot contributes 1 gap to next shot (~220m)
# - So we have roughly equal numbers of small and large spacings
# - Median would be somewhere in between

print(f"\nPossible explanation:")
print(f"  If ~2 receivers per shot:")
print(f"    - {n_shots_estimated} receiver spacings ≈ {actual_spacing:.1f}m each")
print(f"    - {n_shots_estimated-1} shot gaps ≈ {observed_median:.1f}m each")
print(f"    - Median of mixed spacings ≈ {observed_median:.1f}m ✓")
print(f"\n  This would explain the wrong median!")
