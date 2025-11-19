#!/usr/bin/env python3
"""
Test S-Transform inverse reconstruction to verify the normalization fix.
"""

import numpy as np
import sys
from pathlib import Path

# Add denoise_app to path
sys.path.insert(0, str(Path(__file__).parent))

from processors.tf_denoise import stockwell_transform, inverse_stockwell_transform


def test_stransform_roundtrip():
    """Test S-transform forward and inverse reconstruction."""

    print("=" * 70)
    print("S-Transform Reconstruction Test")
    print("=" * 70)

    # Create a test signal with known properties
    n_samples = 1000
    dt = 0.001  # 1ms sampling
    t = np.arange(n_samples) * dt

    # Create multi-frequency signal
    # 50 Hz + 150 Hz + 250 Hz components
    signal = (
        np.sin(2 * np.pi * 50 * t) +
        0.5 * np.sin(2 * np.pi * 150 * t) +
        0.3 * np.sin(2 * np.pi * 250 * t)
    )

    # Calculate input energy
    input_rms = np.sqrt(np.mean(signal**2))
    input_energy = np.sum(signal**2)

    print(f"\n📊 Input Signal:")
    print(f"  Samples: {n_samples}")
    print(f"  Sampling: {1/dt:.0f} Hz")
    print(f"  Frequencies: 50 Hz, 150 Hz, 250 Hz")
    print(f"  RMS: {input_rms:.6f}")
    print(f"  Energy: {input_energy:.6f}")

    # Test 1: Full frequency range (normalized: 0-0.5)
    print(f"\n{'─' * 70}")
    print("TEST 1: Full Frequency Range (all frequencies)")
    print(f"{'─' * 70}")

    fmin_full = None
    fmax_full = None
    S_full, freqs_full = stockwell_transform(
        signal, fmin=fmin_full, fmax=fmax_full
    )

    print(f"  S-transform shape: {S_full.shape}")
    print(f"  Frequencies computed: {len(freqs_full)} (full range)")

    # Inverse transform
    reconstructed_full = inverse_stockwell_transform(S_full, n_samples)

    # Calculate reconstruction quality
    recon_rms = np.sqrt(np.mean(reconstructed_full**2))
    recon_energy = np.sum(reconstructed_full**2)

    # Calculate error
    error = signal - reconstructed_full
    error_rms = np.sqrt(np.mean(error**2))
    energy_ratio = recon_energy / input_energy
    rms_ratio = recon_rms / input_rms

    print(f"\n  Reconstruction:")
    print(f"    RMS: {recon_rms:.6f} (ratio: {rms_ratio:.4f})")
    print(f"    Energy: {recon_energy:.6f} (ratio: {energy_ratio:.4f})")
    print(f"    Error RMS: {error_rms:.6f}")
    print(f"    Error %: {(error_rms/input_rms)*100:.2f}%")

    # Quality check
    if energy_ratio > 0.9 and energy_ratio < 1.1:
        print(f"  ✅ EXCELLENT: Energy conserved within 10%")
    elif energy_ratio > 0.5:
        print(f"  ⚠️  ACCEPTABLE: Energy within 50%")
    else:
        print(f"  ❌ POOR: Lost >50% of energy!")

    # Test 2: Partial frequency range (normalized 0.1-0.3 = 100-300 Hz at 1000Hz sampling)
    print(f"\n{'─' * 70}")
    print("TEST 2: Partial Frequency Range (0.1-0.3 normalized, 100-300 Hz)")
    print(f"{'─' * 70}")

    fmin_partial = 0.1
    fmax_partial = 0.3
    S_partial, freqs_partial = stockwell_transform(
        signal, fmin=fmin_partial, fmax=fmax_partial
    )

    print(f"  S-transform shape: {S_partial.shape}")
    print(f"  Frequencies computed: {len(freqs_partial)} ({fmin_partial:.2f}-{fmax_partial:.2f} normalized)")

    # Inverse transform
    reconstructed_partial = inverse_stockwell_transform(S_partial, n_samples)

    # For partial band, we expect to recover only the 150 Hz and 250 Hz components
    # (50 Hz is outside the band)
    bandlimited_signal = (
        0.5 * np.sin(2 * np.pi * 150 * t) +
        0.3 * np.sin(2 * np.pi * 250 * t)
    )
    bandlimited_rms = np.sqrt(np.mean(bandlimited_signal**2))
    bandlimited_energy = np.sum(bandlimited_signal**2)

    recon_rms_partial = np.sqrt(np.mean(reconstructed_partial**2))
    recon_energy_partial = np.sum(reconstructed_partial**2)

    # Compare to band-limited reference
    error_partial = bandlimited_signal - reconstructed_partial
    error_rms_partial = np.sqrt(np.mean(error_partial**2))
    energy_ratio_partial = recon_energy_partial / bandlimited_energy
    rms_ratio_partial = recon_rms_partial / bandlimited_rms

    print(f"\n  Band-limited reference (150+250 Hz only):")
    print(f"    RMS: {bandlimited_rms:.6f}")
    print(f"    Energy: {bandlimited_energy:.6f}")

    print(f"\n  Reconstruction:")
    print(f"    RMS: {recon_rms_partial:.6f} (ratio: {rms_ratio_partial:.4f})")
    print(f"    Energy: {recon_energy_partial:.6f} (ratio: {energy_ratio_partial:.4f})")
    print(f"    Error RMS: {error_rms_partial:.6f}")
    print(f"    Error %: {(error_rms_partial/bandlimited_rms)*100:.2f}%")

    # Quality check
    if energy_ratio_partial > 0.9 and energy_ratio_partial < 1.1:
        print(f"  ✅ EXCELLENT: Band-limited energy conserved within 10%")
    elif energy_ratio_partial > 0.5:
        print(f"  ⚠️  ACCEPTABLE: Band-limited energy within 50%")
    else:
        print(f"  ❌ POOR: Lost >50% of band-limited energy!")

    # Test 3: Very narrow band (normalized 0.14-0.16 = 140-160 Hz - should recover only 150 Hz component)
    print(f"\n{'─' * 70}")
    print("TEST 3: Narrow Band (0.14-0.16 normalized, 140-160 Hz)")
    print(f"{'─' * 70}")

    fmin_narrow = 0.14
    fmax_narrow = 0.16
    S_narrow, freqs_narrow = stockwell_transform(
        signal, fmin=fmin_narrow, fmax=fmax_narrow
    )

    print(f"  S-transform shape: {S_narrow.shape}")
    print(f"  Frequencies computed: {len(freqs_narrow)} ({fmin_narrow:.2f}-{fmax_narrow:.2f} normalized)")

    # Inverse transform
    reconstructed_narrow = inverse_stockwell_transform(S_narrow, n_samples)

    # Expected: only 150 Hz component
    narrow_signal = 0.5 * np.sin(2 * np.pi * 150 * t)
    narrow_rms = np.sqrt(np.mean(narrow_signal**2))
    narrow_energy = np.sum(narrow_signal**2)

    recon_rms_narrow = np.sqrt(np.mean(reconstructed_narrow**2))
    recon_energy_narrow = np.sum(reconstructed_narrow**2)

    error_narrow = narrow_signal - reconstructed_narrow
    error_rms_narrow = np.sqrt(np.mean(error_narrow**2))
    energy_ratio_narrow = recon_energy_narrow / narrow_energy
    rms_ratio_narrow = recon_rms_narrow / narrow_rms

    print(f"\n  Narrow-band reference (150 Hz only):")
    print(f"    RMS: {narrow_rms:.6f}")
    print(f"    Energy: {narrow_energy:.6f}")

    print(f"\n  Reconstruction:")
    print(f"    RMS: {recon_rms_narrow:.6f} (ratio: {rms_ratio_narrow:.4f})")
    print(f"    Energy: {recon_energy_narrow:.6f} (ratio: {energy_ratio_narrow:.4f})")
    print(f"    Error RMS: {error_rms_narrow:.6f}")
    print(f"    Error %: {(error_rms_narrow/narrow_rms)*100:.2f}%")

    if energy_ratio_narrow > 0.9 and energy_ratio_narrow < 1.1:
        print(f"  ✅ EXCELLENT: Narrow-band energy conserved within 10%")
    elif energy_ratio_narrow > 0.5:
        print(f"  ⚠️  ACCEPTABLE: Narrow-band energy within 50%")
    else:
        print(f"  ❌ POOR: Lost >50% of narrow-band energy!")

    # Final summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    all_passed = True

    print(f"\n  Test 1 (Full spectrum): ", end="")
    if energy_ratio > 0.9 and energy_ratio < 1.1:
        print(f"✅ PASS (energy ratio: {energy_ratio:.4f})")
    else:
        print(f"❌ FAIL (energy ratio: {energy_ratio:.4f})")
        all_passed = False

    print(f"  Test 2 (Partial band): ", end="")
    if energy_ratio_partial > 0.9 and energy_ratio_partial < 1.1:
        print(f"✅ PASS (energy ratio: {energy_ratio_partial:.4f})")
    else:
        print(f"❌ FAIL (energy ratio: {energy_ratio_partial:.4f})")
        all_passed = False

    print(f"  Test 3 (Narrow band): ", end="")
    if energy_ratio_narrow > 0.9 and energy_ratio_narrow < 1.1:
        print(f"✅ PASS (energy ratio: {energy_ratio_narrow:.4f})")
    else:
        print(f"❌ FAIL (energy ratio: {energy_ratio_narrow:.4f})")
        all_passed = False

    print(f"\n{'=' * 70}")
    if all_passed:
        print("🎉 ALL TESTS PASSED - S-transform reconstruction is working correctly!")
    else:
        print("⚠️  SOME TESTS FAILED - Review normalization")
    print(f"{'=' * 70}\n")

    return all_passed


if __name__ == "__main__":
    test_stransform_roundtrip()
