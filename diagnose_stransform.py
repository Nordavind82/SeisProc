#!/usr/bin/env python3
"""
Diagnose S-transform processing to understand why denoising is minimal.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from processors.tf_denoise import stockwell_transform, inverse_stockwell_transform


def test_stransform_roundtrip(signal, label, fmin=None, fmax=None):
    """Test S-transform forward and inverse with given frequency range."""

    print(f"\n{'='*70}")
    print(f"Testing: {label}")
    print(f"{'='*70}")

    n_samples = len(signal)

    # Input statistics
    input_rms = np.sqrt(np.mean(signal**2))
    input_energy = np.sum(signal**2)
    input_min = signal.min()
    input_max = signal.max()

    print(f"\n📊 Input Signal:")
    print(f"   Samples: {n_samples}")
    print(f"   RMS: {input_rms:.6f}")
    print(f"   Energy: {input_energy:.6f}")
    print(f"   Range: [{input_min:.6f}, {input_max:.6f}]")

    # Forward S-transform
    S, freqs = stockwell_transform(signal, fmin=fmin, fmax=fmax)

    print(f"\n📊 S-Transform:")
    print(f"   Shape: {S.shape}")
    print(f"   Frequencies: {len(freqs)} (range: {freqs.min():.4f} to {freqs.max():.4f})")
    print(f"   Magnitude range: [{np.abs(S).min():.6e}, {np.abs(S).max():.6e}]")
    print(f"   Mean magnitude: {np.abs(S).mean():.6e}")

    # Inverse S-transform
    reconstructed = inverse_stockwell_transform(S, n_samples, freq_values=freqs)

    # Reconstruction statistics
    recon_rms = np.sqrt(np.mean(reconstructed**2))
    recon_energy = np.sum(reconstructed**2)
    energy_ratio = recon_energy / input_energy

    error = signal - reconstructed
    error_rms = np.sqrt(np.mean(error**2))
    error_pct = (error_rms / input_rms) * 100

    print(f"\n📊 Reconstruction:")
    print(f"   RMS: {recon_rms:.6f} (input: {input_rms:.6f})")
    print(f"   Energy ratio: {energy_ratio:.4f}", end="")

    if 0.9 <= energy_ratio <= 1.1:
        print(f" ✅ EXCELLENT")
    elif 0.5 <= energy_ratio <= 1.5:
        print(f" ⚠️  Acceptable")
    elif energy_ratio < 0.1:
        print(f" ❌ CRITICAL - Lost {(1-energy_ratio)*100:.1f}% of energy!")
    else:
        print(f" ❌ POOR")

    print(f"   Error RMS: {error_rms:.6f}")
    print(f"   Error %: {error_pct:.2f}%")

    return energy_ratio, error_pct


def test_with_noise(clean_signal, noise_level=0.5):
    """Test S-transform with noisy signal."""

    noise = np.random.randn(len(clean_signal)) * noise_level
    noisy_signal = clean_signal + noise

    print(f"\n{'='*70}")
    print(f"Testing: Noisy Signal (SNR = {1.0/noise_level:.1f})")
    print(f"{'='*70}")

    # Forward transform
    S, freqs = stockwell_transform(noisy_signal, fmin=None, fmax=None)

    # Simple threshold (remove 50% smallest coefficients)
    magnitudes = np.abs(S)
    threshold = np.percentile(magnitudes, 50)

    print(f"\n   Threshold (50th percentile): {threshold:.6e}")
    print(f"   Max magnitude: {magnitudes.max():.6e}")

    # Apply threshold
    S_thresholded = np.where(magnitudes > threshold, S, 0)

    n_total = S.size
    n_kept = np.sum(magnitudes > threshold)
    pct_kept = (n_kept / n_total) * 100

    print(f"   Coefficients kept: {n_kept}/{n_total} ({pct_kept:.1f}%)")

    # Inverse
    denoised = inverse_stockwell_transform(S_thresholded, len(noisy_signal), freq_values=freqs)

    # Statistics
    noisy_rms = np.sqrt(np.mean(noisy_signal**2))
    denoised_rms = np.sqrt(np.mean(denoised**2))
    clean_rms = np.sqrt(np.mean(clean_signal**2))

    noise_removed = noisy_signal - denoised
    noise_removed_rms = np.sqrt(np.mean(noise_removed**2))

    print(f"\n📊 Results:")
    print(f"   Noisy RMS: {noisy_rms:.6f}")
    print(f"   Denoised RMS: {denoised_rms:.6f}")
    print(f"   Clean RMS: {clean_rms:.6f}")
    print(f"   Removed noise RMS: {noise_removed_rms:.6f}")
    print(f"   Expected noise RMS: {noise_level:.6f}")

    # How much signal preserved vs noise removed?
    signal_preserved = denoised_rms / clean_rms
    print(f"\n   Signal preserved: {signal_preserved:.2%}")

    if signal_preserved > 0.8:
        print(f"   ✅ Good - kept most of signal")
    elif signal_preserved < 0.2:
        print(f"   ❌ PROBLEM - Lost most of signal!")
    else:
        print(f"   ⚠️  Moderate signal loss")


if __name__ == "__main__":
    print("=" * 70)
    print("S-Transform Diagnostic Tool")
    print("=" * 70)

    # Test 1: Simple sine wave (full spectrum)
    n = 1000
    t = np.arange(n) * 0.001
    signal1 = np.sin(2 * np.pi * 50 * t)

    ratio1, err1 = test_stransform_roundtrip(
        signal1,
        "Single frequency (50 Hz), Full spectrum",
        fmin=None,
        fmax=None
    )

    # Test 2: Same signal, limited frequency range
    ratio2, err2 = test_stransform_roundtrip(
        signal1,
        "Single frequency (50 Hz), Limited range (0.03-0.15)",
        fmin=0.03,
        fmax=0.15
    )

    # Test 3: Multi-frequency signal (full spectrum)
    signal3 = (
        np.sin(2 * np.pi * 50 * t) +
        0.5 * np.sin(2 * np.pi * 150 * t) +
        0.3 * np.sin(2 * np.pi * 250 * t)
    )

    ratio3, err3 = test_stransform_roundtrip(
        signal3,
        "Multi-frequency (50+150+250 Hz), Full spectrum",
        fmin=None,
        fmax=None
    )

    # Test 4: Multi-frequency, limited range
    ratio4, err4 = test_stransform_roundtrip(
        signal3,
        "Multi-frequency (50+150+250 Hz), Limited range (0.03-0.35)",
        fmin=0.03,
        fmax=0.35
    )

    # Test 5: With noise and thresholding
    test_with_noise(signal3, noise_level=0.5)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    tests = [
        ("Single freq, full spectrum", ratio1, err1),
        ("Single freq, limited range", ratio2, err2),
        ("Multi freq, full spectrum", ratio3, err3),
        ("Multi freq, limited range", ratio4, err4),
    ]

    print(f"\n{'Test':<40} {'Energy Ratio':<15} {'Error %':<10}")
    print(f"{'-'*70}")

    for name, ratio, err in tests:
        status = "✅" if 0.9 <= ratio <= 1.1 else ("⚠️" if 0.5 <= ratio <= 1.5 else "❌")
        print(f"{name:<40} {ratio:>6.4f} {status:>8}  {err:>6.1f}%")

    print(f"\n{'='*70}")

    # Check if limited frequency ranges are the problem
    if ratio2 < 0.5 or ratio4 < 0.5:
        print("⚠️  WARNING: Limited frequency ranges show poor reconstruction!")
        print("   This could explain minimal denoising in practice.")
        print("   The normalization may need adjustment for partial spectra.")
    else:
        print("✅ Reconstruction works for both full and limited frequency ranges")

    print(f"{'='*70}\n")
