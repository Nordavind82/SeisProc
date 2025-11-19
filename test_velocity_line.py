#!/usr/bin/env python3
"""Test velocity line calculation in FK space."""
import numpy as np
from utils.unit_conversion import UnitConverter, format_velocity
from models.app_settings import get_settings

# Simulate the values from screenshot
v_min_display = 922  # ft/s as shown in UI
k_observed = 0.0022  # cycles/m (or cycles/ft?)
f_observed = 170  # Hz (what user sees on plot)

print("=" * 80)
print("VELOCITY LINE DIAGNOSTIC")
print("=" * 80)

# Get current units
settings = get_settings()
current_units = settings.get_spatial_units()
print(f"\nCurrent spatial units: {current_units}")

# Case 1: If v_min is stored internally in m/s
print("\n" + "=" * 80)
print("CASE 1: Velocity stored in m/s (CORRECT)")
print("=" * 80)

if current_units == 'feet':
    # Convert display value to internal (m/s)
    v_internal_ms = UnitConverter.feet_to_meters(v_min_display)
else:
    v_internal_ms = v_min_display

print(f"Display: {v_min_display} ft/s")
print(f"Internal: {v_internal_ms:.2f} m/s")
print(f"Formatted back: {format_velocity(v_internal_ms)}")

# Calculate where line should be at k = 0.0022 cycles/m
k_test = 0.0022  # cycles/m
f_expected = v_internal_ms * k_test

print(f"\nVelocity line calculation:")
print(f"  k = {k_test} cycles/m")
print(f"  f = v × k = {v_internal_ms:.2f} × {k_test} = {f_expected:.3f} Hz")
print(f"  User observes: {f_observed} Hz")
print(f"  Ratio: {f_observed / f_expected:.1f}×")

# Case 2: If v_min is stored in ft/s (BUG!)
print("\n" + "=" * 80)
print("CASE 2: Velocity stored in ft/s (BUG - wrong units)")
print("=" * 80)

v_wrong_fts = v_min_display  # stored in ft/s instead of m/s
f_wrong = v_wrong_fts * k_test  # ft/s × cycles/m = WRONG!

print(f"If v stored as: {v_wrong_fts} ft/s (should be m/s)")
print(f"And multiplied with k in cycles/m:")
print(f"  f = {v_wrong_fts} × {k_test} = {f_wrong:.3f} Hz")
print(f"  User observes: {f_observed} Hz")
print(f"  Ratio: {f_observed / f_wrong:.1f}×")

# Case 3: What if k is in different units?
print("\n" + "=" * 80)
print("CASE 3: Check if k units are wrong")
print("=" * 80)

# If wavenumbers are in cycles/foot instead of cycles/m
k_if_feet = k_observed * UnitConverter.meters_to_feet()
print(f"\nIf k were in cycles/foot:")
print(f"  k = {k_test} cycles/m = {k_if_feet:.6f} cycles/ft")

f_with_ft_k = v_min_display * k_if_feet  # ft/s × cycles/ft = Hz (correct!)
print(f"  f = {v_min_display} ft/s × {k_if_feet:.6f} cycles/ft = {f_with_ft_k:.3f} Hz")
print(f"  User observes: {f_observed} Hz")
print(f"  Ratio: {f_observed / f_with_ft_k:.1f}×")

# Try to find what's causing 170 Hz
print("\n" + "=" * 80)
print("REVERSE ENGINEER: What causes f = 170 Hz?")
print("=" * 80)

implied_v = f_observed / k_test
print(f"\nAt k = {k_test}, f = {f_observed} Hz")
print(f"Implied velocity: v = f/k = {implied_v:.0f} units")
print(f"In m/s: {implied_v:.0f} m/s = {implied_v * 3.28:.0f} ft/s")
print(f"In ft/s treated as m/s: {implied_v / 3.28:.0f} ft/s → {implied_v:.0f} 'display'")

# Check if there's a systematic error
print(f"\nRatio of implied to expected:")
print(f"  {implied_v:.0f} / {v_internal_ms:.0f} = {implied_v / v_internal_ms:.1f}×")
print(f"  {implied_v:.0f} / {v_min_display:.0f} = {implied_v / v_min_display:.1f}×")
