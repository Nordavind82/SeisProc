#!/usr/bin/env python3
"""
Test script to verify wavenumber calculation with actual parameters.
"""
import numpy as np
from scipy import fft

# Parameters from the screenshot
trace_spacing = 219.9  # meters
n_traces = 497  # from screenshot "497 traces"

# Calculate wavenumbers using the same method as FK filter
wavenumbers = fft.fftfreq(n_traces, trace_spacing)

print("=" * 60)
print("FK Wavenumber Calculation Test")
print("=" * 60)
print(f"\nInput Parameters:")
print(f"  Number of traces: {n_traces}")
print(f"  Trace spacing: {trace_spacing} m")

print(f"\nCalculated Wavenumbers:")
print(f"  Min wavenumber: {wavenumbers.min():.6f} cycles/m = {wavenumbers.min()*1000:.3f} mcycles/m")
print(f"  Max wavenumber: {wavenumbers.max():.6f} cycles/m = {wavenumbers.max()*1000:.3f} mcycles/m")
print(f"  Nyquist wavenumber: {1/(2*trace_spacing):.6f} cycles/m = {1000/(2*trace_spacing):.3f} mcycles/m")

print(f"\nWavenumber range in FK plot:")
print(f"  Should be: ±{wavenumbers.max():.6f} cycles/m")
print(f"  Should be: ±{wavenumbers.max()*1000:.3f} mcycles/m")

print(f"\nExpected FK plot range (from screenshot): ±15 mcycles/m")
print(f"Calculated FK plot range: ±{wavenumbers.max()*1000:.3f} mcycles/m")

if abs(wavenumbers.max()*1000 - 15) > 1:
    print("\n⚠️  WARNING: Mismatch detected!")
    print(f"   Expected: ±15 mcycles/m")
    print(f"   Calculated: ±{wavenumbers.max()*1000:.3f} mcycles/m")
    print(f"   Difference: {abs(wavenumbers.max()*1000 - 15):.3f} mcycles/m")
else:
    print("\n✓ Wavenumber calculation matches screenshot!")

# Test velocity filter boundaries
print("\n" + "=" * 60)
print("Velocity Filter Boundary Test")
print("=" * 60)

v_min = 184  # m/s
v_max = 6000  # m/s

test_frequencies = [50, 100, 150, 200, 250]

print(f"\nFor v_min = {v_min} m/s:")
for f in test_frequencies:
    k = f / v_min
    print(f"  At f={f:3d} Hz: k = {k:.4f} cycles/m = {k*1000:.1f} mcycles/m")

print(f"\nFor v_max = {v_max} m/s:")
for f in test_frequencies:
    k = f / v_max
    print(f"  At f={f:3d} Hz: k = {k:.6f} cycles/m = {k*1000:.3f} mcycles/m")

print(f"\nFK plot range: ±{wavenumbers.max()*1000:.3f} mcycles/m")
print(f"\nConclusion:")
k_vmin_at_100Hz = 100 / v_min * 1000
k_vmax_at_100Hz = 100 / v_max * 1000
print(f"  v_min={v_min} m/s boundary at 100 Hz: {k_vmin_at_100Hz:.1f} mcycles/m")
print(f"  v_max={v_max} m/s boundary at 100 Hz: {k_vmax_at_100Hz:.3f} mcycles/m")
print(f"  FK plot range: ±{wavenumbers.max()*1000:.3f} mcycles/m")

if k_vmin_at_100Hz > wavenumbers.max()*1000:
    print(f"\n⚠️  v_min boundary is {k_vmin_at_100Hz/wavenumbers.max()/1000:.1f}x BEYOND plot range!")
    print(f"   This means the v_min filter is essentially rejecting ALL visible data!")
