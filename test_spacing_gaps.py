#!/usr/bin/env python3
"""
Test script to detect large gaps in receiver spacing that indicate sub-gather boundaries.
"""
import numpy as np
import pandas as pd
from utils.trace_spacing import calculate_trace_spacing_with_stats, apply_segy_scalar

# This would be run on your actual data
# For now, let's create a test to show what SHOULD happen

def analyze_spacing_gaps(headers_df: pd.DataFrame, threshold_multiplier: float = 3.0):
    """
    Detect large gaps in receiver spacing that might indicate sub-gather boundaries.

    Args:
        headers_df: DataFrame with trace headers
        threshold_multiplier: Gap is considered large if > threshold_multiplier * median_spacing

    Returns:
        Analysis results
    """
    print("=" * 70)
    print("RECEIVER SPACING GAP ANALYSIS")
    print("=" * 70)

    # Try to get receiver coordinates
    coord_candidates = ['receiver_x', 'ReceiverX', 'GroupX', 'receiver_x1']
    scalar_candidates = ['scalar_coord', 'scalco']

    coords_raw = None
    coord_header = None
    scalar = 1.0

    for coord_name in coord_candidates:
        if coord_name in headers_df.columns:
            coords_raw = headers_df[coord_name].values
            coord_header = coord_name

            # Get scalar
            for scalar_name in scalar_candidates:
                if scalar_name in headers_df.columns:
                    scalar_val = headers_df[scalar_name].iloc[0]
                    if scalar_val != 0:
                        scalar = scalar_val
                    break
            break

    if coords_raw is None:
        print("No receiver coordinate headers found!")
        return

    print(f"\nUsing: {coord_header}")
    print(f"Number of traces: {len(coords_raw)}")
    print(f"SEGY scalar: {scalar}")

    # Apply scalar
    coords = apply_segy_scalar(coords_raw, scalar)

    print(f"\nCoordinate range:")
    print(f"  Min: {coords.min():.2f} m")
    print(f"  Max: {coords.max():.2f} m")
    print(f"  Span: {coords.max() - coords.min():.2f} m")

    # Calculate all spacings
    spacings = np.abs(np.diff(coords))
    spacings_nonzero = spacings[spacings > 0]

    if len(spacings_nonzero) == 0:
        print("\nAll coordinates are the same!")
        return

    median_spacing = np.median(spacings_nonzero)
    mean_spacing = np.mean(spacings_nonzero)
    std_spacing = np.std(spacings_nonzero)

    print(f"\nSpacing statistics (all traces):")
    print(f"  Median: {median_spacing:.2f} m")
    print(f"  Mean: {mean_spacing:.2f} m")
    print(f"  Std: {std_spacing:.2f} m")
    print(f"  Min: {spacings_nonzero.min():.2f} m")
    print(f"  Max: {spacings_nonzero.max():.2f} m")

    # Detect large gaps
    gap_threshold = median_spacing * threshold_multiplier
    large_gaps = np.where(spacings > gap_threshold)[0]

    print(f"\nGap detection (threshold = {threshold_multiplier}x median = {gap_threshold:.2f} m):")
    print(f"  Found {len(large_gaps)} large gaps")

    if len(large_gaps) > 0:
        print(f"\n  Large gaps detected at traces:")
        for i, gap_idx in enumerate(large_gaps[:10]):  # Show first 10
            gap_size = spacings[gap_idx]
            print(f"    Trace {gap_idx:3d} → {gap_idx+1:3d}: gap = {gap_size:7.2f} m ({gap_size/median_spacing:.1f}x median)")

        if len(large_gaps) > 10:
            print(f"    ... and {len(large_gaps) - 10} more gaps")

        # Estimate spacing within groups (excluding large gaps)
        normal_spacings = spacings[spacings <= gap_threshold]
        if len(normal_spacings) > 0:
            within_group_spacing = np.median(normal_spacings)
            print(f"\n  Estimated spacing WITHIN sub-gathers: {within_group_spacing:.2f} m")
            print(f"  Estimated spacing BETWEEN sub-gathers: {np.median(spacings[large_gaps]):.2f} m")

            print(f"\n  *** RECOMMENDATION ***")
            print(f"  Current (wrong) spacing: {median_spacing:.2f} m (includes gaps)")
            print(f"  Correct spacing should be: {within_group_spacing:.2f} m (within groups only)")
            print(f"  Factor difference: {median_spacing/within_group_spacing:.2f}x")

            # Expected FK range
            k_wrong = 1000 / (2 * median_spacing)  # mcycles/m
            k_correct = 1000 / (2 * within_group_spacing)  # mcycles/m

            print(f"\n  FK wavenumber range:")
            print(f"    With current spacing: ±{k_wrong:.3f} mcycles/m")
            print(f"    With correct spacing: ±{k_correct:.3f} mcycles/m")
    else:
        print(f"\n  No large gaps detected - spacing appears uniform")
        print(f"  This gather may not need sub-gather splitting")

    return {
        'median_spacing': median_spacing,
        'large_gaps': large_gaps,
        'gap_threshold': gap_threshold
    }

# Test with simulated data matching user's case
print("SIMULATED TEST (matching user's data pattern)")
print("=" * 70)

# Simulate 11 shot points, each with 45 receivers at 33m spacing
# Shot points separated by ~220m
n_shots = 11
receivers_per_shot = 45
receiver_spacing = 33.0  # m
shot_spacing = 220.0  # m

coords_list = []
for shot in range(n_shots):
    shot_start = shot * shot_spacing
    for rx in range(receivers_per_shot):
        coords_list.append(shot_start + rx * receiver_spacing)

coords_simulated = np.array(coords_list)

# Create fake dataframe
df_test = pd.DataFrame({
    'receiver_x': coords_simulated,
    'scalar_coord': [-1] * len(coords_simulated)  # Coordinates already in meters
})

print(f"\nSimulated data:")
print(f"  {n_shots} shot points")
print(f"  {receivers_per_shot} receivers per shot")
print(f"  {receiver_spacing} m receiver spacing")
print(f"  {shot_spacing} m shot point spacing")
print(f"  Total traces: {len(coords_simulated)}")

result = analyze_spacing_gaps(df_test, threshold_multiplier=2.0)
