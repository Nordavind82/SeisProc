#!/usr/bin/env python3
"""
Find the exact normalization factor needed for S-transform inverse.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from processors.tf_denoise import stockwell_transform


# Test with multiple signals to find pattern
print("=" * 70)
print("Finding S-Transform Normalization Factor")
print("=" * 70)

for test_size in [50, 100, 200, 500]:
    t = np.arange(test_size) * 0.001
    signal = np.sin(2 * np.pi * 50 * t)

    # Forward S-transform
    S, freqs = stockwell_transform(signal, fmin=None, fmax=None)

    # Raw inverse (sum and take real)
    time_series = np.sum(S, axis=0)
    reconstructed_raw = time_series.real

    # Calculate energies
    input_energy = np.sum(signal**2)
    recon_energy = np.sum(reconstructed_raw**2)
    ratio = recon_energy / input_energy

    # What factor do we need?
    needed_factor = np.sqrt(ratio)

    n_samples = len(signal)
    n_freqs = S.shape[0]

    print(f"\nTest size: {test_size} samples")
    print(f"  n_freqs: {n_freqs}")
    print(f"  Raw energy ratio: {ratio:.6f}")
    print(f"  Factor needed to divide: {needed_factor:.6f}")
    print(f"  Relation to n_samples: {needed_factor / n_samples:.6f}")
    print(f"  Relation to n_freqs: {needed_factor / n_freqs:.6f}")
    print(f"  Relation to sqrt(n_samples): {needed_factor / np.sqrt(n_samples):.6f}")
    print(f"  n_freqs / n_samples: {n_freqs / n_samples:.6f}")

    # Check if the factor is related to n_freqs
    if abs(needed_factor - n_freqs) < 1:
        print(f"  ✅ Factor ≈ n_freqs ({n_freqs})")
    elif abs(needed_factor / np.sqrt(n_samples) - 1) < 0.1:
        print(f"  ✅ Factor ≈ sqrt(n_samples) ({np.sqrt(n_samples):.2f})")

print(f"\n{'=' * 70}")
print("CONCLUSION:")
print(f"{'=' * 70}")
print("\nIt appears the normalization should be by sqrt(n_freqs * 2)")
print("Or possibly by sqrt(n_samples / 2)")
print("Let me check the exact relationship...")

# Final check with the most common case
n_samples = 1000
t = np.arange(n_samples) * 0.001
signal = np.sin(2 * np.pi * 50 * t)

S, freqs = stockwell_transform(signal, fmin=None, fmax=None)
time_series = np.sum(S, axis=0)
reconstructed_raw = time_series.real

input_energy = np.sum(signal**2)
recon_energy = np.sum(reconstructed_raw**2)
ratio = recon_energy / input_energy
needed_factor = np.sqrt(ratio)

n_freqs = S.shape[0]

print(f"\nFinal test (n_samples={n_samples}):")
print(f"  n_freqs: {n_freqs}")
print(f"  Raw energy ratio: {ratio:.6f}")
print(f"  Needed division factor: {needed_factor:.6f}")

# Try various formulas
formulas = {
    "n_freqs": n_freqs,
    "sqrt(n_freqs * 2)": np.sqrt(n_freqs * 2),
    "sqrt(n_samples / 2)": np.sqrt(n_samples / 2),
    "sqrt(n_samples)": np.sqrt(n_samples),
    "n_samples / n_freqs": n_samples / n_freqs,
}

print(f"\nChecking formulas:")
for name, value in formulas.items():
    error = abs(value - needed_factor)
    print(f"  {name:25s} = {value:8.3f}  (error: {error:.3f})")

    if error < 1:
        print(f"    ✅ MATCH! Normalization should be: reconstructed / {name}")
