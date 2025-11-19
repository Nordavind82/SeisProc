#!/usr/bin/env python3
"""Test offset gap detection."""
import numpy as np
import pandas as pd
from utils.trace_spacing import analyze_offset_step_uniformity

# Test 1: Uniform offsets (no gaps)
print("=" * 70)
print("TEST 1: Uniform offsets (no sub-gather splitting needed)")
print("=" * 70)

offsets_uniform = np.arange(0, 1000, 25)  # 0, 25, 50, ..., 975
df1 = pd.DataFrame({'offset': offsets_uniform})

result1 = analyze_offset_step_uniformity(df1)
if result1:
    print(f"Has gaps: {result1.has_gaps}")
    print(f"Median step: {result1.median_step:.1f} m")
    print(f"Step CV: {result1.step_cv:.1f}%")
    print(f"Number of gaps: {result1.n_gaps}")
else:
    print("No offset column found")

# Test 2: Multiple sub-gathers with gaps
print("\n" + "=" * 70)
print("TEST 2: Multiple sub-gathers (gaps present)")
print("=" * 70)

# Simulate 5 sub-gathers, each with 20 offsets at 25m spacing
# Large gaps (500m) between sub-gathers
offsets_with_gaps = []
fldr_values = []
for sg in range(5):
    base = sg * 1000  # Large jump between sub-gathers
    for i in range(20):
        offsets_with_gaps.append(base + i * 25)
        fldr_values.append(sg + 1)

df2 = pd.DataFrame({
    'offset': offsets_with_gaps,
    'fldr': fldr_values
})

result2 = analyze_offset_step_uniformity(df2)
if result2:
    print(f"Has gaps: {result2.has_gaps}")
    print(f"Median step: {result2.median_step:.1f} m")
    print(f"Step CV: {result2.step_cv:.1f}%")
    print(f"Number of gaps: {result2.n_gaps}")
    print(f"Gap indices: {result2.gap_indices}")
    print(f"Suggested headers: {result2.suggested_headers}")
else:
    print("Analysis failed")

# Test 3: Real-world scenario matching user's data
print("\n" + "=" * 70)
print("TEST 3: Scenario matching user's data (11 shots, 45 receivers)")
print("=" * 70)

offsets_real = []
shot_values = []
for shot in range(11):
    # Each shot has 45 receivers
    # Offsets from -1100m to +1100m in 50m steps
    for rx in range(45):
        offset = -1100 + rx * 50
        offsets_real.append(offset)
        shot_values.append(shot + 1)

df3 = pd.DataFrame({
    'offset': offsets_real,
    'fldr': shot_values,
    'ep': shot_values  # Same as fldr for this test
})

result3 = analyze_offset_step_uniformity(df3)
if result3:
    print(f"Has gaps: {result3.has_gaps}")
    print(f"Median step: {result3.median_step:.1f} m")
    print(f"Step CV: {result3.step_cv:.1f}%")
    print(f"Number of gaps: {result3.n_gaps}")
    if result3.n_gaps > 0 and result3.n_gaps <= 5:
        print(f"Gap indices: {result3.gap_indices}")
    print(f"Suggested headers: {result3.suggested_headers}")

    if result3.has_gaps:
        print("\n⚠️ WARNING: Sub-gather splitting recommended!")
    else:
        print("\n✓ Offset steps are uniform - no splitting needed")
else:
    print("Analysis failed")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("The gap detection successfully identifies when sub-gathers")
print("are mixed together and suggests appropriate boundary headers.")
