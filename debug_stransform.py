#!/usr/bin/env python3
"""
Debug S-transform to understand the normalization issue.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from processors.tf_denoise import stockwell_transform, inverse_stockwell_transform


# Simple test signal
n_samples = 100
t = np.arange(n_samples) * 0.001
signal = np.sin(2 * np.pi * 50 * t)  # 50 Hz sine wave

print("=" * 70)
print("S-Transform Debug - Single Frequency")
print("=" * 70)
print(f"\nInput signal: 50 Hz sine wave")
print(f"  Samples: {n_samples}")
print(f"  RMS: {np.sqrt(np.mean(signal**2)):.6f}")
print(f"  Energy: {np.sum(signal**2):.6f}")
print(f"  Min/Max: {signal.min():.6f} / {signal.max():.6f}")

# Forward S-transform (full spectrum)
S, freqs = stockwell_transform(signal, fmin=None, fmax=None)

print(f"\nS-transform:")
print(f"  Shape: {S.shape}")
print(f"  Frequencies: {len(freqs)}")

# Check S-transform properties
S_magnitude = np.abs(S)
print(f"\n  S-transform magnitude:")
print(f"    Min: {S_magnitude.min():.6e}")
print(f"    Max: {S_magnitude.max():.6e}")
print(f"    Mean: {S_magnitude.mean():.6e}")
print(f"    Sum: {S_magnitude.sum():.6e}")

# Inverse with different normalization factors
print(f"\n{'─' * 70}")
print("Testing different normalization factors:")
print(f"{'─' * 70}")

# Sum across frequencies
time_series = np.sum(S, axis=0)
reconstructed_raw = time_series.real

print(f"\nRaw reconstruction (no normalization):")
print(f"  RMS: {np.sqrt(np.mean(reconstructed_raw**2)):.6e}")
print(f"  Energy: {np.sum(reconstructed_raw**2):.6e}")
print(f"  Min/Max: {reconstructed_raw.min():.6e} / {reconstructed_raw.max():.6e}")
print(f"  Ratio to input: {np.sum(reconstructed_raw**2) / np.sum(signal**2):.6e}")

# Try different normalizations
normalizations = {
    "/ n_samples": n_samples,
    "/ n_freqs": S.shape[0],
    "/ sqrt(n_samples)": np.sqrt(n_samples),
    "/ (n_freqs * n_samples)": S.shape[0] * n_samples,
    "* 2 / n_samples": n_samples / 2,
    "* 2 / n_freqs": S.shape[0] / 2,
    "No normalization": 1.0,
}

best_ratio = 0
best_norm = None
best_signal = None

for name, factor in normalizations.items():
    if "No" in name:
        recon = reconstructed_raw
    elif "*" in name:
        recon = reconstructed_raw * factor
    else:
        recon = reconstructed_raw / factor

    energy_ratio = np.sum(recon**2) / np.sum(signal**2)
    rms_ratio = np.sqrt(np.mean(recon**2)) / np.sqrt(np.mean(signal**2))

    print(f"\n{name}:")
    print(f"  Factor: {factor}")
    print(f"  Energy ratio: {energy_ratio:.6e}")
    print(f"  RMS ratio: {rms_ratio:.6e}")

    if abs(energy_ratio - 1.0) < abs(best_ratio - 1.0):
        best_ratio = energy_ratio
        best_norm = name
        best_signal = recon

print(f"\n{'=' * 70}")
print(f"BEST NORMALIZATION: {best_norm}")
print(f"  Energy ratio: {best_ratio:.6e}")
print(f"  Error from ideal (1.0): {abs(best_ratio - 1.0):.6e}")
print(f"{'=' * 70}")
