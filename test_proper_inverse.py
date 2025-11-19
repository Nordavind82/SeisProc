#!/usr/bin/env python3
"""
Test proper S-transform inverse based on Stockwell's original paper.

According to the S-transform literature, the inverse should be:
x(t) = ∫ S(t,f) df  (integrated over all frequencies)

For discrete implementation with only positive frequencies:
x[n] = Σ_f S[f, n]  (but might need factor of 2 to account for negative freqs)
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from processors.tf_denoise import stockwell_transform


def inverse_stockwell_v2(S, n_samples, only_positive_freqs=True):
    """
    Alternative inverse S-transform implementation.

    The S-transform at each time point is essentially a frequency spectrum.
    To invert, we sum across frequencies.

    For positive frequencies only, we need factor of 2 (except DC and Nyquist).
    """
    if S.shape[0] == 0:
        return np.zeros(n_samples)

    # Sum across frequency axis
    time_series = np.sum(S, axis=0)

    # Take real part
    reconstructed = time_series.real

    # If we only computed positive frequencies (not including negative),
    # we need to multiply by 2 to account for the missing negative frequencies
    # (except for DC component if included)
    if only_positive_freqs:
        reconstructed = reconstructed * 2.0

    return reconstructed[:n_samples]


def inverse_stockwell_v3(S, n_samples, freq_values):
    """
    Weighted inverse based on frequency values.
    """
    if S.shape[0] == 0:
        return np.zeros(n_samples)

    # Weight each frequency by its value (higher frequencies get more weight)
    # This accounts for the Gaussian window scaling
    weights = np.abs(freq_values).reshape(-1, 1)
    weights = np.where(weights == 0, 1.0, weights)  # Avoid division by zero

    weighted_S = S * weights

    time_series = np.sum(weighted_S, axis=0)
    reconstructed = time_series.real

    # Normalize
    reconstructed = reconstructed * 2.0 / n_samples

    return reconstructed[:n_samples]


# Test signal
n_samples = 100
t = np.arange(n_samples) * 0.001
signal = np.sin(2 * np.pi * 50 * t)

print("=" * 70)
print("Testing S-Transform Inverse Variations")
print("=" * 70)
print(f"\nInput: 50 Hz sine wave, {n_samples} samples")
print(f"  RMS: {np.sqrt(np.mean(signal**2)):.6f}")
print(f"  Energy: {np.sum(signal**2):.6f}")

# Forward transform
S, freqs = stockwell_transform(signal, fmin=None, fmax=None)
print(f"\nS-transform shape: {S.shape}")
print(f"  Frequencies: {len(freqs)}")
print(f"  Freq range: {freqs.min():.4f} to {freqs.max():.4f}")

# Test inverse version 2 (with factor of 2)
print(f"\n{'─' * 70}")
print("Version 2: Simple sum × 2 (account for negative freqs)")
print(f"{'─' * 70}")

recon_v2 = inverse_stockwell_v2(S, n_samples, only_positive_freqs=True)
energy_v2 = np.sum(recon_v2**2)
ratio_v2 = energy_v2 / np.sum(signal**2)

print(f"  Energy ratio: {ratio_v2:.6f}")
print(f"  RMS ratio: {np.sqrt(np.mean(recon_v2**2)) / np.sqrt(np.mean(signal**2)):.6f}")

if abs(ratio_v2 - 1.0) < 0.1:
    print(f"  ✅ GOOD! Within 10% of input energy")
elif abs(ratio_v2 - 1.0) < 0.5:
    print(f"  ⚠️  Acceptable (within 50%)")
else:
    print(f"  ❌ Poor reconstruction")

# Test inverse version 3 (weighted)
print(f"\n{'─' * 70}")
print("Version 3: Frequency-weighted sum")
print(f"{'─' * 70}")

recon_v3 = inverse_stockwell_v3(S, n_samples, freqs)
energy_v3 = np.sum(recon_v3**2)
ratio_v3 = energy_v3 / np.sum(signal**2)

print(f"  Energy ratio: {ratio_v3:.6f}")
print(f"  RMS ratio: {np.sqrt(np.mean(recon_v3**2)) / np.sqrt(np.mean(signal**2)):.6f}")

if abs(ratio_v3 - 1.0) < 0.1:
    print(f"  ✅ GOOD! Within 10% of input energy")
elif abs(ratio_v3 - 1.0) < 0.5:
    print(f"  ⚠️  Acceptable (within 50%)")
else:
    print(f"  ❌ Poor reconstruction")

# Try dividing by sqrt(raw_ratio) that we found earlier
print(f"\n{'─' * 70}")
print("Version 4: Empirical normalization")
print(f"{'─' * 70}")

time_series = np.sum(S, axis=0)
recon_raw = time_series.real
raw_ratio = np.sum(recon_raw**2) / np.sum(signal**2)
empirical_factor = np.sqrt(raw_ratio)

recon_v4 = recon_raw / empirical_factor
energy_v4 = np.sum(recon_v4**2)
ratio_v4 = energy_v4 / np.sum(signal**2)

print(f"  Raw ratio: {raw_ratio:.6f}")
print(f"  Empirical factor: {empirical_factor:.6f}")
print(f"  Energy ratio: {ratio_v4:.6f}")
print(f"  RMS ratio: {np.sqrt(np.mean(recon_v4**2)) / np.sqrt(np.mean(signal**2)):.6f}")

if abs(ratio_v4 - 1.0) < 0.1:
    print(f"  ✅ GOOD! Within 10% of input energy")
elif abs(ratio_v4 - 1.0) < 0.5:
    print(f"  ⚠️  Acceptable (within 50%)")
else:
    print(f"  ❌ Poor reconstruction")

# Summary
print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")

results = [
    ("Version 2 (×2)", ratio_v2),
    ("Version 3 (weighted)", ratio_v3),
    ("Version 4 (empirical)", ratio_v4),
]

best = min(results, key=lambda x: abs(x[1] - 1.0))
print(f"\nBest result: {best[0]}")
print(f"  Energy ratio: {best[1]:.6f}")
print(f"  Error from ideal: {abs(best[1] - 1.0)*100:.2f}%")
