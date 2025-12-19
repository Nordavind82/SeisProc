#!/usr/bin/env python3
"""
Minimal test to verify STFT fix is working.
"""

import numpy as np
import sys
import os
import importlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_minimal():
    """Minimal test with forced module reload."""
    print("=" * 60)
    print("MINIMAL STFT TEST")
    print("=" * 60)

    # Force reload of the metal module
    try:
        # Clear any cached imports
        mods_to_remove = [k for k in sys.modules if 'seismic_metal' in k.lower()]
        for mod in mods_to_remove:
            del sys.modules[mod]
    except:
        pass

    from processors.kernel_backend import get_metal_module, is_metal_available

    if not is_metal_available():
        print("Metal not available")
        return

    metal = get_metal_module()

    # Simplest possible test: single trace, short signal
    n_samples = 128
    n_traces = 1
    t = np.linspace(0, 2 * np.pi, n_samples)

    # Use a signal that starts at non-zero
    traces = np.cos(t).reshape(n_samples, 1).astype(np.float32)

    print(f"\nInput signal: cos(t)")
    print(f"  Shape: {traces.shape}")
    print(f"  First sample: {traces[0, 0]:.6f} (should be 1.0)")
    print(f"  First 5: {traces[:5, 0]}")

    # Test with minimal parameters
    result, _ = metal.stft_denoise(
        traces,
        nperseg=32,  # Smaller window
        noverlap=24,
        aperture=1,
        threshold_k=10000.0,  # Very high - no thresholding
        fmin=0.0,
        fmax=0.0,
        sample_rate=500.0
    )

    print(f"\nOutput:")
    print(f"  First sample: {result[0, 0]:.6f} (should be ~1.0)")
    print(f"  First 5: {result[:5, 0]}")

    error = np.max(np.abs(traces - result))
    first_error = np.abs(traces[0, 0] - result[0, 0])

    print(f"\nMax error: {error:.6f}")
    print(f"First sample error: {first_error:.6f}")

    # Check if first sample is being zeroed
    if result[0, 0] < 0.5:
        print("\nWARNING: First sample appears to be zeroed - padding fix not working!")
    else:
        print("\nFirst sample preserved - padding is working!")


if __name__ == "__main__":
    test_minimal()
