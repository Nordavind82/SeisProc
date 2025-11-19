#!/usr/bin/env python3
"""
Verification script to confirm TF-Denoise outputs signal model (not noise model).

Expected behavior:
- Output RMS should be 70-95% of input RMS
- Output energy < Input energy (some noise removed)
- Processed + Difference ≈ Input (energy conservation)
"""

import numpy as np


def verify_signal_output(input_data, processed_data, difference_data):
    """
    Verify that processed output is signal model.

    Args:
        input_data: Original noisy data
        processed_data: TF-Denoise output
        difference_data: Removed noise
    """
    # Compute RMS values
    input_rms = np.sqrt(np.mean(input_data**2))
    processed_rms = np.sqrt(np.mean(processed_data**2))
    difference_rms = np.sqrt(np.mean(difference_data**2))

    # Compute energies
    input_energy = np.sum(input_data**2)
    processed_energy = np.sum(processed_data**2)
    difference_energy = np.sum(difference_data**2)

    # Verify reconstruction
    reconstructed = processed_data + difference_data
    reconstruction_error = np.sqrt(np.mean((input_data - reconstructed)**2))

    print("=" * 60)
    print("TF-Denoise Output Verification")
    print("=" * 60)

    print("\n1. RMS Analysis:")
    print(f"   Input RMS:      {input_rms:.6f}")
    print(f"   Processed RMS:  {processed_rms:.6f}")
    print(f"   Difference RMS: {difference_rms:.6f}")
    print(f"   Ratio (Out/In): {processed_rms/input_rms:.2%}")

    if 0.70 <= processed_rms/input_rms <= 0.95:
        print("   ✓ PASS: Output RMS is 70-95% of input (signal preserved)")
    else:
        print("   ✗ WARNING: Unexpected RMS ratio")

    print("\n2. Energy Analysis:")
    print(f"   Input Energy:      {input_energy:.2e}")
    print(f"   Processed Energy:  {processed_energy:.2e}")
    print(f"   Removed Energy:    {difference_energy:.2e}")
    print(f"   Removed:           {100*difference_energy/input_energy:.1f}%")

    if processed_energy < input_energy:
        print("   ✓ PASS: Output energy < Input energy (noise removed)")
    else:
        print("   ✗ FAIL: Output energy >= Input energy")

    print("\n3. Reconstruction Check:")
    print(f"   Reconstruction error: {reconstruction_error:.2e}")
    print(f"   Relative error:       {reconstruction_error/input_rms:.2%}")

    if reconstruction_error/input_rms < 0.01:
        print("   ✓ PASS: Processed + Difference ≈ Input")
    else:
        print("   ✗ WARNING: Reconstruction error high")

    print("\n" + "=" * 60)
    print("Conclusion:")
    if processed_energy < input_energy:
        print("✓ OUTPUT IS SIGNAL MODEL (denoised data)")
        print("  - Processed data has less energy than input")
        print("  - Difference contains the removed noise")
    else:
        print("✗ OUTPUT APPEARS TO BE NOISE MODEL")
        print("  - Check if panels are swapped")
    print("=" * 60)

    return {
        'input_rms': input_rms,
        'processed_rms': processed_rms,
        'difference_rms': difference_rms,
        'rms_ratio': processed_rms/input_rms,
        'energy_removed_pct': 100*difference_energy/input_energy,
        'reconstruction_error': reconstruction_error
    }


if __name__ == "__main__":
    # Example usage with synthetic data
    print("Example with synthetic data:\n")

    # Create synthetic signal + noise
    n_samples = 1000
    t = np.linspace(0, 1, n_samples)

    # Signal: sum of sinusoids
    signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 25 * t)

    # Noise: white noise
    noise = 0.3 * np.random.randn(n_samples)

    # Input: signal + noise
    input_data = signal + noise

    # Simulate denoising (80% noise removed)
    processed_data = signal + 0.2 * noise
    difference_data = 0.8 * noise

    results = verify_signal_output(input_data, processed_data, difference_data)

    print("\nTo use with actual TF-Denoise output:")
    print("  from verify_tf_output import verify_signal_output")
    print("  verify_signal_output(input_traces, processed_traces, difference_traces)")
