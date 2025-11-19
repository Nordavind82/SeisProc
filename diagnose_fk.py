#!/usr/bin/env python3
"""
Diagnostic script to check FK spectrum calculation with real data.
Run this to see the actual wavenumbers being calculated.
"""
import numpy as np
from scipy import fft

print("=" * 70)
print("FK SPECTRUM DIAGNOSTIC")
print("=" * 70)

# Test with different trace spacings to find the multiplier
expected_k_max = 15e-3  # 15 mcycles/m from screenshot = 0.015 cycles/m
n_traces = 497

print(f"\nExpected from screenshot:")
print(f"  FK plot wavenumber range: ±15 mcycles/m = ±{expected_k_max} cycles/m")
print(f"  Number of traces: {n_traces}")

# What trace spacing would give us k_max = 15 mcycles/m?
# k_max = (n_traces/2) / (n_traces * dx)
# k_max = 1 / (2 * dx)
# dx = 1 / (2 * k_max)

implied_dx = 1.0 / (2.0 * expected_k_max)
print(f"\nImplied trace spacing to get k_max = {expected_k_max} cycles/m:")
print(f"  dx = 1/(2*k_max) = {implied_dx:.2f} m")

# What we're told from the screenshot
reported_dx = 219.9  # meters

print(f"\nReported trace spacing: {reported_dx:.2f} m")

ratio = reported_dx / implied_dx
print(f"\nRatio: reported/implied = {ratio:.3f}")

print(f"\nPossible explanations:")
print(f"1. Trace spacing is actually {implied_dx:.2f} m, not {reported_dx:.2f} m")
print(f"2. There's a unit conversion error (factor of {ratio:.2f})")
print(f"3. The FK plot axis labels are wrong")

# Check if it could be a meters vs other unit issue
print(f"\nUnit check:")
print(f"  {reported_dx:.2f} m = {reported_dx*100:.1f} cm")
print(f"  {reported_dx:.2f} m = {reported_dx*3.281:.1f} feet")
print(f"  {implied_dx:.2f} m = {implied_dx*100:.1f} cm")
print(f"  {implied_dx:.2f} m = {implied_dx*3.281:.1f} feet")

# Check what happens with fft.fftfreq
print("\n" + "=" * 70)
print("FFT.FFTFREQ TEST")
print("=" * 70)

for dx in [implied_dx, reported_dx]:
    k = fft.fftfreq(n_traces, dx)
    print(f"\nfft.fftfreq({n_traces}, dx={dx:.2f}m):")
    print(f"  k_min = {k.min():.6f} cycles/m = {k.min()*1000:.3f} mcycles/m")
    print(f"  k_max = {k.max():.6f} cycles/m = {k.max()*1000:.3f} mcycles/m")
    print(f"  Nyquist = {1/(2*dx):.6f} cycles/m = {1000/(2*dx):.3f} mcycles/m")

# Check if maybe there's an issue with how the coordinates are being scaled
print("\n" + "=" * 70)
print("COORDINATE SCALING CHECK")
print("=" * 70)

# If receiver coordinates have a scalar issue
print("\nIf raw coordinates are in different units:")
scalars_to_test = [-1, -10, -100, -1000, 1, 10, 100, 1000]
for scalar in scalars_to_test:
    if scalar < 0:
        # Divide by abs(scalar)
        effective_spacing = reported_dx / abs(scalar)
    else:
        # Multiply
        effective_spacing = reported_dx * scalar

    if 20 < effective_spacing < 50:  # Reasonable range for seismic
        k_max_test = 1 / (2 * effective_spacing)
        print(f"  Scalar {scalar:5d}: spacing = {effective_spacing:6.2f} m → k_max = {k_max_test*1000:6.3f} mcycles/m")
        if abs(k_max_test * 1000 - 15) < 1:
            print(f"    *** MATCH! This scalar would give the observed FK range ***")
