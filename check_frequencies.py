#!/usr/bin/env python3
"""
Check what frequencies the S-transform actually returns.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from processors.tf_denoise import stockwell_transform

# Simple test
n_samples = 100
signal = np.sin(2 * np.pi * 0.05 * np.arange(n_samples))  # Normalized freq 0.05

S, freqs = stockwell_transform(signal, fmin=None, fmax=None)

print("=" * 70)
print("S-Transform Frequency Analysis")
print("=" * 70)
print(f"\nSignal: {n_samples} samples")
print(f"S-transform shape: {S.shape}")
print(f"\nFrequencies returned ({len(freqs)} values):")
print(f"  Min: {freqs.min():.6f}")
print(f"  Max: {freqs.max():.6f}")
print(f"  First 10: {freqs[:10]}")
print(f"  Last 10: {freqs[-10:]}")

# Check if they're actually positive
if np.any(freqs < 0):
    print(f"\n⚠️  WARNING: Contains negative frequencies!")
    n_negative = np.sum(freqs < 0)
    n_positive = np.sum(freqs > 0)
    n_zero = np.sum(freqs == 0)
    print(f"  Negative: {n_negative}")
    print(f"  Zero: {n_zero}")
    print(f"  Positive: {n_positive}")
else:
    print(f"\n✅ All positive frequencies")

# Check the normalization factor pattern
print(f"\n{'─' * 70}")
print("Empirical Factor Analysis")
print(f"{'─' * 70}")

for n in [50, 100, 200, 500, 1000]:
    signal = np.sin(2 * np.pi * 0.05 * np.arange(n))
    S, freqs = stockwell_transform(signal, fmin=None, fmax=None)

    time_series = np.sum(S, axis=0)
    recon_raw = time_series.real

    input_energy = np.sum(signal**2)
    recon_energy = np.sum(recon_raw**2)
    raw_ratio = recon_energy / input_energy
    empirical_factor = np.sqrt(raw_ratio)

    print(f"\nn_samples={n:4d}, n_freqs={S.shape[0]:4d}")
    print(f"  Raw energy ratio: {raw_ratio:10.6f}")
    print(f"  Empirical factor: {empirical_factor:10.6f}")
    print(f"  Factor / n_freqs: {empirical_factor / S.shape[0]:10.6f}")
    print(f"  Factor / sqrt(n): {empirical_factor / np.sqrt(n):10.6f}")
    print(f"  n_freqs / sqrt(n): {S.shape[0] / np.sqrt(n):10.6f}")

    # Check if empirical_factor ≈ sqrt(2 * n_freqs)
    theoretical = np.sqrt(2 * S.shape[0])
    error = abs(empirical_factor - theoretical) / empirical_factor * 100
    print(f"  sqrt(2 * n_freqs): {theoretical:10.6f}  (error: {error:.2f}%)")

    # Check if it's constant
    # (skip ratio check for now)
