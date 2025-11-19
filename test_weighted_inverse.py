#!/usr/bin/env python3
"""
Test the new frequency-weighted inverse S-transform.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from processors.tf_denoise import stockwell_transform, inverse_stockwell_transform

print("=" * 70)
print("Testing Frequency-Weighted Inverse S-Transform")
print("=" * 70)

# Test with different frequency signals
test_cases = [
    (0.05, "50 Hz (low freq)"),
    (0.10, "100 Hz (mid-low)"),
    (0.20, "200 Hz (mid-high)"),
    (0.30, "300 Hz (high freq)"),
]

n_samples = 1000

for test_freq, label in test_cases:
    signal = np.sin(2 * np.pi * test_freq * np.arange(n_samples))

    # Forward transform
    S, freqs = stockwell_transform(signal, fmin=None, fmax=None)

    # Inverse with frequency weighting
    reconstructed = inverse_stockwell_transform(S, n_samples, freq_values=freqs)

    # Calculate quality
    input_energy = np.sum(signal**2)
    recon_energy = np.sum(reconstructed**2)
    energy_ratio = recon_energy / input_energy

    error = signal - reconstructed
    error_rms = np.sqrt(np.mean(error**2))
    error_pct = (error_rms / np.sqrt(np.mean(signal**2))) * 100

    print(f"\n{label} (normalized freq: {test_freq}):")
    print(f"  Energy ratio: {energy_ratio:.4f}", end="")

    if 0.9 <= energy_ratio <= 1.1:
        print(f"  ✅ EXCELLENT")
    elif 0.5 <= energy_ratio <= 1.5:
        print(f"  ⚠️  Acceptable")
    else:
        print(f"  ❌ POOR")

    print(f"  Error: {error_pct:.2f}%")

# Test with multi-frequency signal
print(f"\n{'─' * 70}")
print("Multi-frequency test (50+150+250 Hz):")
print(f"{'─' * 70}")

t = np.arange(n_samples) * 0.001
signal = (
    np.sin(2 * np.pi * 50 * t) +
    0.5 * np.sin(2 * np.pi * 150 * t) +
    0.3 * np.sin(2 * np.pi * 250 * t)
)

S, freqs = stockwell_transform(signal, fmin=None, fmax=None)
reconstructed = inverse_stockwell_transform(S, n_samples, freq_values=freqs)

input_energy = np.sum(signal**2)
recon_energy = np.sum(reconstructed**2)
energy_ratio = recon_energy / input_energy

error = signal - reconstructed
error_rms = np.sqrt(np.mean(error**2))
error_pct = (error_rms / np.sqrt(np.mean(signal**2))) * 100

print(f"\n  Input RMS: {np.sqrt(np.mean(signal**2)):.6f}")
print(f"  Recon RMS: {np.sqrt(np.mean(reconstructed**2)):.6f}")
print(f"  Energy ratio: {energy_ratio:.4f}", end="")

if 0.9 <= energy_ratio <= 1.1:
    print(f"  ✅ EXCELLENT")
elif 0.5 <= energy_ratio <= 1.5:
    print(f"  ⚠️  Acceptable")
else:
    print(f"  ❌ POOR")

print(f"  Reconstruction error: {error_pct:.2f}%")

print(f"\n{'=' * 70}")
if energy_ratio >= 0.9 and energy_ratio <= 1.1:
    print("✅ Frequency-weighted inverse is working correctly!")
else:
    print("⚠️  Inverse needs further tuning")
print(f"{'=' * 70}\n")
