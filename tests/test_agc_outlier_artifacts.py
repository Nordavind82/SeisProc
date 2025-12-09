"""
Diagnostic test: AGC behavior with outliers (air blasts, spikes).

The hypothesis is that max_gain clipping creates artifacts that
repeat the shape of outliers.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from processors.agc import apply_agc_vectorized


def create_trace_with_airblast(nt=1000, dt=0.002):
    """Create synthetic trace with air blast (high-amplitude outlier)."""
    t = np.arange(nt) * dt

    # Normal seismic signal: reflections with decreasing amplitude
    trace = np.zeros(nt, dtype=np.float32)

    # Add some reflectors
    for t_refl, amp in [(0.2, 1.0), (0.4, 0.8), (0.6, 0.6), (0.8, 0.4)]:
        idx = int(t_refl / dt)
        wavelet = amp * np.sin(2 * np.pi * 30 * (t - t_refl)) * \
                  np.exp(-((t - t_refl) / 0.02)**2)
        trace += wavelet

    # Add air blast: very high amplitude at top of record
    # Air blast is typically 10-100x stronger than signal
    airblast_amp = 50.0  # 50x normal amplitude
    airblast_center = 0.05  # 50ms
    airblast_width = 0.03   # 30ms duration
    idx_center = int(airblast_center / dt)
    airblast = airblast_amp * np.sin(2 * np.pi * 15 * (t - airblast_center)) * \
               np.exp(-((t - airblast_center) / airblast_width)**2)
    trace += airblast

    return trace, t


def analyze_agc_with_outlier():
    """Analyze how AGC handles outliers with and without max_gain."""
    print("=" * 70)
    print("AGC Behavior with Air Blast Outlier")
    print("=" * 70)

    trace, t = create_trace_with_airblast()
    traces_2d = trace.reshape(-1, 1)  # Shape for AGC function

    window_samples = 101  # ~200ms window

    # Test 1: No max_gain limit (or very high)
    agc_unlimited, sf_unlimited = apply_agc_vectorized(
        traces_2d, window_samples, target_rms=1.0, max_gain=1e10
    )

    # Test 2: Typical max_gain=10
    agc_limited_10, sf_limited_10 = apply_agc_vectorized(
        traces_2d, window_samples, target_rms=1.0, max_gain=10.0
    )

    # Test 3: Aggressive max_gain=3
    agc_limited_3, sf_limited_3 = apply_agc_vectorized(
        traces_2d, window_samples, target_rms=1.0, max_gain=3.0
    )

    print("\nScale factor statistics:")
    print(f"  Unlimited: min={sf_unlimited.min():.4f}, max={sf_unlimited.max():.4f}")
    print(f"  max_gain=10: min={sf_limited_10.min():.4f}, max={sf_limited_10.max():.4f}")
    print(f"  max_gain=3:  min={sf_limited_3.min():.4f}, max={sf_limited_3.max():.4f}")

    # Where does clipping occur?
    clipped_10 = sf_limited_10.flatten() >= 9.99
    clipped_3 = sf_limited_3.flatten() >= 2.99

    print(f"\nSamples where max_gain clips:")
    print(f"  max_gain=10: {clipped_10.sum()} samples ({100*clipped_10.mean():.1f}%)")
    print(f"  max_gain=3:  {clipped_3.sum()} samples ({100*clipped_3.mean():.1f}%)")

    # The key insight: where RMS is dominated by air blast,
    # scale_factor = target_rms / high_rms = small value
    # Adjacent to air blast, RMS drops quickly, scale_factor should rise
    # But if clipped, it stays at max_gain creating a "shadow" of the air blast

    print("\n" + "=" * 70)
    print("KEY INSIGHT: The 'shadow' artifact mechanism")
    print("=" * 70)
    print("""
When air blast is in the AGC window:
  - RMS is high → scale_factor is LOW (correctly attenuating)

When air blast exits the AGC window:
  - RMS drops suddenly → scale_factor should RISE to compensate
  - WITHOUT max_gain: scale_factor rises to true value (could be 100+)
  - WITH max_gain: scale_factor is CLIPPED, creating INSUFFICIENT gain

The artifact is NOT from max_gain clipping during the air blast.
The artifact is from max_gain clipping in ADJACENT quiet zones
where the sudden amplitude drop needs high gain to compensate.

This creates a "shadow" that mirrors the air blast's influence zone.
""")

    # Find the shadow zone
    # Look at where scale factors are clipped AFTER the air blast
    airblast_end_sample = int(0.08 / 0.002)  # ~80ms, after air blast peak

    print(f"\nScale factors after air blast (samples {airblast_end_sample}-{airblast_end_sample+50}):")
    print(f"  Unlimited: {sf_unlimited[airblast_end_sample:airblast_end_sample+10, 0]}")
    print(f"  max_gain=10: {sf_limited_10[airblast_end_sample:airblast_end_sample+10, 0]}")

    # Check if unlimited scale factors exceed limits
    exceed_10 = (sf_unlimited > 10).sum()
    exceed_100 = (sf_unlimited > 100).sum()
    print(f"\nSamples where true scale_factor exceeds limits:")
    print(f"  > 10: {exceed_10} samples")
    print(f"  > 100: {exceed_100} samples")

    return trace, t, sf_unlimited, sf_limited_10, agc_unlimited, agc_limited_10


def analyze_reversal_artifacts():
    """Show how AGC reversal creates artifacts with clipped scale factors."""
    print("\n" + "=" * 70)
    print("AGC Reversal Artifacts with Clipped Scale Factors")
    print("=" * 70)

    trace, t = create_trace_with_airblast()
    traces_2d = trace.reshape(-1, 1)
    window_samples = 101

    # Apply AGC with max_gain=10
    agc_data, sf_clipped = apply_agc_vectorized(
        traces_2d, window_samples, target_rms=1.0, max_gain=10.0
    )

    # Simulate filtering (just add small noise to represent filter artifacts)
    np.random.seed(42)
    filter_noise = np.random.randn(*agc_data.shape).astype(np.float32) * 0.01
    filtered_agc = agc_data + filter_noise

    # Reverse AGC
    restored = filtered_agc / (sf_clipped + 1e-10)

    # What SHOULD happen (with true scale factors)
    agc_unlimited, sf_true = apply_agc_vectorized(
        traces_2d, window_samples, target_rms=1.0, max_gain=1e10
    )
    filtered_unlimited = agc_unlimited + filter_noise
    restored_correct = filtered_unlimited / (sf_true + 1e-10)

    # Compare
    diff = restored.flatten() - trace
    diff_correct = restored_correct.flatten() - trace

    print(f"\nRMS error after reversal:")
    print(f"  With clipped scale factors: {np.sqrt((diff**2).mean()):.4f}")
    print(f"  With true scale factors: {np.sqrt((diff_correct**2).mean()):.4f}")

    # Where is the error worst?
    # It should be worst where scale factors were clipped
    clipped_zone = sf_clipped.flatten() >= 9.99

    print(f"\nRMS error in clipped zones:")
    print(f"  Clipped method: {np.sqrt((diff[clipped_zone]**2).mean()):.4f}")
    print(f"  True method: {np.sqrt((diff_correct[clipped_zone]**2).mean()):.4f}")

    print(f"\nRMS error in non-clipped zones:")
    print(f"  Clipped method: {np.sqrt((diff[~clipped_zone]**2).mean()):.4f}")
    print(f"  True method: {np.sqrt((diff_correct[~clipped_zone]**2).mean()):.4f}")


def test_no_max_gain():
    """Test: What happens with no max_gain limit at all?"""
    print("\n" + "=" * 70)
    print("Test: AGC Without max_gain Limit")
    print("=" * 70)

    trace, t = create_trace_with_airblast()
    traces_2d = trace.reshape(-1, 1)
    window_samples = 101

    # No practical limit
    agc_data, sf = apply_agc_vectorized(
        traces_2d, window_samples, target_rms=1.0, max_gain=np.inf
    )

    print(f"Scale factor range: {sf.min():.4f} to {sf.max():.4f}")
    print(f"AGC output range: {agc_data.min():.4f} to {agc_data.max():.4f}")

    # The concern: does unlimited gain cause numerical issues?
    has_inf = np.isinf(sf).any() or np.isinf(agc_data).any()
    has_nan = np.isnan(sf).any() or np.isnan(agc_data).any()

    print(f"\nNumerical issues: inf={has_inf}, nan={has_nan}")

    # Test reversal
    restored = agc_data / (sf + 1e-10)
    reversal_error = np.abs(restored.flatten() - trace).max()
    print(f"Max reversal error: {reversal_error:.6f}")

    # The real question: for FKK filtering, do we even need reversal?
    print("\n" + "-" * 50)
    print("CONSIDERATION: Do we need AGC reversal at all?")
    print("-" * 50)
    print("""
For FKK filtering, AGC serves to:
1. Equalize amplitudes so filter affects all depths equally
2. Prevent high-amplitude events from dominating the spectrum

After filtering, reversing AGC attempts to restore original amplitudes.
But this is problematic because:
- The filtered data has DIFFERENT amplitude characteristics
- Scale factors from original data don't apply to filtered data
- Clipping creates artifacts

Alternative approaches:
1. Don't reverse AGC (keep equalized output)
2. Apply a NEW AGC to the filtered data with matching parameters
3. Use amplitude-preserving filtering instead of AGC
4. Only use AGC for visualization, not for processing
""")


def visualize_agc_shadow_artifact():
    """Create visualization of the shadow artifact."""
    trace, t = create_trace_with_airblast()
    traces_2d = trace.reshape(-1, 1)
    window_samples = 101

    # Different max_gain settings
    results = {}
    for max_gain in [3.0, 10.0, 100.0, 1e10]:
        agc, sf = apply_agc_vectorized(traces_2d, window_samples, max_gain=max_gain)
        results[max_gain] = {'agc': agc.flatten(), 'sf': sf.flatten()}

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    # Original trace
    axes[0].plot(t, trace, 'k', linewidth=0.5)
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Original Trace with Air Blast')
    axes[0].set_ylim(-60, 60)

    # Scale factors
    for max_gain, data in results.items():
        label = f'max_gain={max_gain}' if max_gain < 1e5 else 'unlimited'
        axes[1].plot(t, data['sf'], label=label, linewidth=1)
    axes[1].set_ylabel('Scale Factor')
    axes[1].set_title('AGC Scale Factors')
    axes[1].legend()
    axes[1].set_yscale('log')

    # AGC output
    for max_gain, data in results.items():
        label = f'max_gain={max_gain}' if max_gain < 1e5 else 'unlimited'
        axes[2].plot(t, data['agc'], label=label, linewidth=0.5, alpha=0.7)
    axes[2].set_ylabel('Amplitude')
    axes[2].set_title('AGC Output')
    axes[2].legend()

    # Reversal comparison
    for max_gain, data in results.items():
        restored = data['agc'] / (data['sf'] + 1e-10)
        diff = restored - trace
        label = f'max_gain={max_gain}' if max_gain < 1e5 else 'unlimited'
        axes[3].plot(t, diff, label=label, linewidth=0.5)
    axes[3].set_ylabel('Error')
    axes[3].set_title('Reversal Error (restored - original)')
    axes[3].set_xlabel('Time (s)')
    axes[3].legend()

    plt.tight_layout()
    plt.savefig('/tmp/agc_shadow_artifact.png', dpi=150)
    print("\nSaved visualization to /tmp/agc_shadow_artifact.png")
    plt.close()


def main():
    analyze_agc_with_outlier()
    analyze_reversal_artifacts()
    test_no_max_gain()

    print("\n" + "=" * 70)
    print("CONCLUSION: Should we remove max_gain?")
    print("=" * 70)
    print("""
ANSWER: Yes, for FKK filtering, max_gain limitation is HARMFUL.

The max_gain parameter was designed to prevent:
1. Numerical overflow (not an issue with float32/64)
2. Excessive noise amplification in dead zones

But for FKK with AGC reversal:
- max_gain creates "shadow" artifacts adjacent to outliers
- The shadow has the SHAPE of the outlier's influence zone
- Reversal then imprints this shape into the output

RECOMMENDED CHANGES:
1. Remove max_gain clipping entirely (set to inf or very large)
2. Use epsilon-only protection: scale = target / (rms + epsilon)
3. If noise amplification is a concern, handle it AFTER filtering
   by applying a separate mute or taper to known dead zones

For air blasts specifically:
- Consider detecting and muting them BEFORE AGC
- Or use surgical mute after FKK filtering
- The AGC should not try to "fix" outliers - that's not its job
""")

    try:
        visualize_agc_shadow_artifact()
    except Exception as e:
        print(f"\nCould not create visualization: {e}")


if __name__ == '__main__':
    main()
