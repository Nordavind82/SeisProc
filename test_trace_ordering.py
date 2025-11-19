#!/usr/bin/env python3
"""
Test different trace ordering scenarios.
"""
import numpy as np

print("=" * 70)
print("TRACE ORDERING ANALYSIS")
print("=" * 70)

# Setup: 11 shots, 45 receivers each
n_shots = 11
n_receivers = 45
rx_spacing = 33.0
shot_spacing = 220.0

print(f"\nSetup:")
print(f"  {n_shots} shot points")
print(f"  {n_receivers} receivers per shot")
print(f"  {rx_spacing}m receiver spacing")
print(f"  {shot_spacing}m between shots")

# Scenario 1: Ordered by RECEIVER (all shot 1, then all shot 2, etc.)
print(f"\n{'='*70}")
print("SCENARIO 1: Ordered by SHOT (typical for shot gathers)")
print('='*70)
print("Trace ordering: [S1R1, S1R2, ..., S1R45, S2R1, S2R2, ..., S11R45]")

coords_by_shot = []
for shot in range(n_shots):
    shot_start = shot * shot_spacing
    for rx in range(n_receivers):
        coords_by_shot.append(shot_start + rx * rx_spacing)

coords_by_shot = np.array(coords_by_shot)
spacings_by_shot = np.abs(np.diff(coords_by_shot))
median_by_shot = np.median(spacings_by_shot)

print(f"\nMedian spacing: {median_by_shot:.1f} m")
print(f"FK range: ±{1000/(2*median_by_shot):.3f} mcycles/m")

# Scenario 2: Ordered by RECEIVER (all receivers at position 1, then all at position 2, etc.)
print(f"\n{'='*70}")
print("SCENARIO 2: Ordered by RECEIVER POSITION (unusual but possible)")
print('='*70)
print("Trace ordering: [S1R1, S2R1, ..., S11R1, S1R2, S2R2, ..., S11R45]")

coords_by_receiver = []
for rx in range(n_receivers):
    for shot in range(n_shots):
        shot_start = shot * shot_spacing
        coords_by_receiver.append(shot_start + rx * rx_spacing)

coords_by_receiver = np.array(coords_by_receiver)
spacings_by_receiver = np.abs(np.diff(coords_by_receiver))
median_by_receiver = np.median(spacings_by_receiver)

print(f"\nMedian spacing: {median_by_receiver:.1f} m")
print(f"FK range: ±{1000/(2*median_by_receiver):.3f} mcycles/m")

# Show spacing distribution for Scenario 2
unique, counts = np.unique(spacings_by_receiver.astype(int), return_counts=True)
print(f"\nSpacing distribution:")
for spacing, count in zip(unique[:10], counts[:10]):  # Show first 10
    print(f"  {spacing:4d} m: {count:3d} occurrences")

if median_by_receiver > 200:
    print(f"\n⚠️  MATCH FOUND!")
    print(f"   If traces are ordered by receiver position, median ≈ {median_by_receiver:.1f} m")
    print(f"   This matches your observed {219.9:.1f} m!")

# Scenario 3: Check actual trace coordinate progression
print(f"\n{'='*70}")
print("DIAGNOSIS")
print('='*70)

print(f"\nYour data:")
print(f"  Reported median spacing: 219.9 m")
print(f"  Observed FK range: ±15 mcycles/m")
print(f"  Expected FK range (from 219.9m): ±{1000/(2*219.9):.3f} mcycles/m ❌")
print(f"  Actual spacing needed for ±15 mcycles/m: {1000/(2*15):.1f} m")

print(f"\nConclusion:")
print(f"  The FK spectrum is using the CORRECT spacing (33.3 m)")
print(f"  But the median calculation is WRONG (219.9 m)")
print(f"  This means:")
print(f"    - Traces might be ordered by receiver position (scenario 2)")
print(f"    - OR median is being calculated incorrectly")
print(f"    - OR there's a sub-gather/geometry issue")

print(f"\n{'='*70}")
print("RECOMMENDATION")
print('='*70)
print(f"\n1. Enable 'Split gather by header changes'")
print(f"2. Select boundary header (try: 'inline', 'fldr', 'shot_point', 'ep')")
print(f"3. This will separate data into proper shot gathers")
print(f"4. Each sub-gather will have correct ~33m spacing")
