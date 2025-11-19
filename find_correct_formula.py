#!/usr/bin/env python3
"""
Find the correct normalization formula for S-transform inverse.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from processors.tf_denoise import stockwell_transform

print("=" * 70)
print("Finding S-Transform Normalization Formula")
print("=" * 70)

# Collect data
data_points = []

for n in [50, 100, 200, 500, 1000]:
    signal = np.sin(2 * np.pi * 0.05 * np.arange(n))
    S, freqs = stockwell_transform(signal, fmin=None, fmax=None)

    time_series = np.sum(S, axis=0)
    recon_raw = time_series.real

    input_energy = np.sum(signal**2)
    recon_energy = np.sum(recon_raw**2)
    raw_ratio = recon_energy / input_energy
    needed_factor = np.sqrt(raw_ratio)

    n_freqs = S.shape[0]

    data_points.append((n, n_freqs, needed_factor, raw_ratio))

    print(f"\nn={n:4d}, n_freqs={n_freqs:4d}, factor={needed_factor:8.4f}")

# Try to find a formula
print(f"\n{'=' * 70}")
print("Testing Candidate Formulas:")
print(f"{'=' * 70}")

formulas = [
    ("sqrt(n_freqs)", lambda n, nf: np.sqrt(nf)),
    ("sqrt(n)", lambda n, nf: np.sqrt(n)),
    ("sqrt(n/2)", lambda n, nf: np.sqrt(n/2)),
    ("sqrt(2*nf)", lambda n, nf: np.sqrt(2*nf)),
    ("n_freqs/sqrt(n)", lambda n, nf: nf/np.sqrt(n)),
    ("sqrt(n_freqs/2)", lambda n, nf: np.sqrt(nf/2)),
    ("sqrt(n*nf)/n", lambda n, nf: np.sqrt(n*nf)/n),
]

best_formula = None
best_error = float('inf')

for formula_name, formula_func in formulas:
    total_error = 0
    max_error = 0

    print(f"\n{formula_name}:")
    for n, nf, true_factor, _ in data_points:
        pred_factor = formula_func(n, nf)
        error = abs(pred_factor - true_factor) / true_factor * 100
        total_error += error
        max_error = max(max_error, error)
        print(f"  n={n:4d}: pred={pred_factor:8.4f}, true={true_factor:8.4f}, error={error:6.2f}%")

    avg_error = total_error / len(data_points)
    print(f"  Avg error: {avg_error:.2f}%, Max error: {max_error:.2f}%")

    if avg_error < best_error:
        best_error = avg_error
        best_formula = formula_name

print(f"\n{'=' * 70}")
print(f"Best formula: {best_formula}")
print(f"Average error: {best_error:.2f}%")
print(f"{'=' * 70}")

# If no formula is good enough, maybe it depends on the signal?
# Let me check if the factor changes with different frequencies
print(f"\n{'=' * 70}")
print("Testing if factor depends on signal frequency:")
print(f"{'=' * 70}")

n = 1000
for test_freq in [0.01, 0.05, 0.1, 0.2, 0.3]:
    signal = np.sin(2 * np.pi * test_freq * np.arange(n))
    S, freqs = stockwell_transform(signal, fmin=None, fmax=None)

    time_series = np.sum(S, axis=0)
    recon_raw = time_series.real

    input_energy = np.sum(signal**2)
    recon_energy = np.sum(recon_raw**2)
    raw_ratio = recon_energy / input_energy
    needed_factor = np.sqrt(raw_ratio)

    print(f"  Freq={test_freq:.2f}: factor={needed_factor:8.4f}, ratio={raw_ratio:8.4f}")
