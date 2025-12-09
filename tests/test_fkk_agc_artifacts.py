"""
Diagnostic tests for FKK filter top-of-record artifacts.

Isolates AGC and padding interactions to identify root cause of artifacts.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from processors.agc import apply_agc_vectorized
from processors.fkk_filter_gpu import (
    apply_agc_3d, remove_agc_3d, apply_temporal_taper,
    pad_copy_temporal, pad_copy_3d, get_auto_pad_size
)
from models.seismic_volume import SeismicVolume
from models.fkk_config import FKKConfig


def create_test_volume(nt=200, nx=20, ny=20, dt=0.002, dx=25.0, dy=25.0):
    """Create synthetic volume with known characteristics."""
    t = np.arange(nt) * dt

    # Create data with:
    # - Low amplitude at top (first 20 samples)
    # - Strong reflector at t=0.1s (sample 50)
    # - Medium amplitude elsewhere
    data = np.zeros((nt, nx, ny), dtype=np.float32)

    # Background noise
    np.random.seed(42)
    data += np.random.randn(nt, nx, ny).astype(np.float32) * 0.1

    # Low amplitude zone at top
    data[:20, :, :] *= 0.01  # Very low amplitude

    # Strong reflector
    wavelet_center = 50
    wavelet = np.sin(2 * np.pi * 30 * (t - t[wavelet_center])) * \
              np.exp(-((t - t[wavelet_center]) / 0.01)**2)
    for ix in range(nx):
        for iy in range(ny):
            data[:, ix, iy] += wavelet * 5.0  # Strong event

    return SeismicVolume(data=data, dt=dt, dx=dx, dy=dy)


def test_agc_scale_factors_at_top():
    """Test 1: Examine AGC scale factors at top of record."""
    print("=" * 60)
    print("TEST 1: AGC Scale Factors at Top of Record (Adaptive Epsilon)")
    print("=" * 60)

    vol = create_test_volume()
    data = vol.data.copy()

    # Apply AGC with adaptive epsilon (no max_gain clipping)
    window_samples = 51  # ~100ms window
    agc_data, scale_factors = apply_agc_3d(data, window_samples)

    # Analyze scale factors
    sf_trace = scale_factors[:, 10, 10]  # Central trace

    print(f"Scale factors at top 20 samples: {sf_trace[:20]}")
    print(f"Scale factors at samples 40-60: {sf_trace[40:60]}")
    print(f"Max scale factor: {sf_trace.max():.2f}")
    print(f"Min scale factor: {sf_trace.min():.2f}")
    print(f"Scale factor at top (sample 0): {sf_trace[0]:.2f}")
    print(f"Scale factor at sample 50 (reflector): {sf_trace[50]:.2f}")

    # With adaptive epsilon, scale factors should now be reasonable
    # (not clipped to an arbitrary max, but naturally limited by epsilon)
    if sf_trace[:20].max() > 100.0:
        print("\n⚠️  Note: High AGC gain at top of record (expected for low-amplitude zones)")
    else:
        print("\n✓ Scale factors at top are reasonable with adaptive epsilon")

    return scale_factors


def test_agc_reversal_amplifies_artifacts():
    """Test 2: Show AGC reversal behavior with adaptive epsilon."""
    print("\n" + "=" * 60)
    print("TEST 2: AGC Reversal Effect on Small Perturbations (Adaptive Epsilon)")
    print("=" * 60)

    vol = create_test_volume()
    data = vol.data.copy()

    # Apply AGC with adaptive epsilon
    window_samples = 51
    agc_data, scale_factors = apply_agc_3d(data, window_samples)

    # Simulate filter adding small uniform artifact
    artifact_amplitude = 0.05  # Small artifact from FFT edge effects
    agc_data_with_artifact = agc_data.copy()
    agc_data_with_artifact[:30, :, :] += artifact_amplitude  # Artifact at top

    # Remove AGC
    restored_clean = remove_agc_3d(agc_data, scale_factors)
    restored_artifact = remove_agc_3d(agc_data_with_artifact, scale_factors)

    # Compare artifact appearance after AGC reversal
    diff = restored_artifact - restored_clean

    print(f"Input artifact amplitude: {artifact_amplitude:.4f}")
    print(f"Artifact after AGC reversal at top (sample 5): {diff[5, 10, 10]:.4f}")
    print(f"Artifact after AGC reversal at sample 50: {diff[50, 10, 10]:.4f}")

    # With adaptive epsilon, these should be more proportional
    if abs(diff[5, 10, 10]) > 1e-10:
        print(f"Attenuation factor at top: {diff[5, 10, 10] / artifact_amplitude:.2f}x")
    if abs(diff[50, 10, 10]) > 1e-10:
        print(f"Attenuation factor at sample 50: {diff[50, 10, 10] / artifact_amplitude:.2f}x")

    top_amp = np.abs(diff[:20, :, :]).mean()
    mid_amp = np.abs(diff[40:60, :, :]).mean()
    print(f"\nMean artifact at top (0-20): {top_amp:.4f}")
    print(f"Mean artifact at mid (40-60): {mid_amp:.4f}")
    if mid_amp > 1e-10:
        print(f"Relative ratio: {top_amp/mid_amp:.2f}x")

    if mid_amp > 1e-10 and top_amp / mid_amp > 2.0:
        print("\n⚠️  CONFIRMED: AGC reversal amplifies top artifacts!")


def test_temporal_taper_placement():
    """Test 3: Check if temporal taper is applied before or after AGC."""
    print("\n" + "=" * 60)
    print("TEST 3: Temporal Taper vs AGC Ordering")
    print("=" * 60)

    vol = create_test_volume()
    data = vol.data.copy()

    # Current order: AGC first, then taper
    window_samples = 51
    agc_data, sf = apply_agc_3d(data, window_samples)
    tapered_after_agc = apply_temporal_taper(agc_data, taper_top=30, taper_bottom=0)

    # Alternative: Taper first, then AGC
    tapered_data = apply_temporal_taper(data, taper_top=30, taper_bottom=0)
    agc_after_taper, sf2 = apply_agc_3d(tapered_data, window_samples)

    trace1 = tapered_after_agc[:50, 10, 10]
    trace2 = agc_after_taper[:50, 10, 10]

    print("Current approach (AGC then taper):")
    print(f"  Samples 0-10: {trace1[:10]}")
    print(f"  RMS top 30: {np.sqrt((trace1[:30]**2).mean()):.4f}")

    print("\nAlternative (taper then AGC):")
    print(f"  Samples 0-10: {trace2[:10]}")
    print(f"  RMS top 30: {np.sqrt((trace2[:30]**2).mean()):.4f}")

    print("\nNote: When AGC is applied first, the taper sees equalized amplitudes.")
    print("When taper is applied first, AGC sees tapered (low) amplitudes at top,")
    print("leading to even higher gain factors and worse reversal artifacts.")


def test_pad_copy_temporal_continuity():
    """Test 4: Check pad_copy_temporal creates smooth transition."""
    print("\n" + "=" * 60)
    print("TEST 4: Temporal Pad-Copy Continuity")
    print("=" * 60)

    vol = create_test_volume()
    data = vol.data.copy()

    # Apply temporal pad_copy
    pad_top = 20
    pad_bottom = 10
    padded = pad_copy_temporal(data, pad_top, pad_bottom)

    trace_orig = data[:30, 10, 10]
    trace_padded = padded[:50, 10, 10]  # Includes pad_top + some original

    print(f"Original shape: {data.shape}, Padded shape: {padded.shape}")
    print(f"\nOriginal trace first 5 samples: {trace_orig[:5]}")
    print(f"Padded trace samples around junction (pad_top={pad_top}):")
    print(f"  Samples {pad_top-5} to {pad_top+5}: {trace_padded[pad_top-5:pad_top+5]}")

    # Check discontinuity
    jump = abs(trace_padded[pad_top] - trace_padded[pad_top-1])
    print(f"\nJump at junction (sample {pad_top-1} to {pad_top}): {jump:.4f}")

    if jump > 0.5:
        print("⚠️  Large discontinuity at pad-data boundary!")


def test_full_fkk_with_agc_isolation():
    """Test 5: Run FKK with and without AGC, compare top artifacts."""
    print("\n" + "=" * 60)
    print("TEST 5: FKK Filter With vs Without AGC")
    print("=" * 60)

    try:
        from processors.fkk_filter_gpu import get_fkk_filter
    except ImportError as e:
        print(f"Could not import FKK filter: {e}")
        return

    vol = create_test_volume()
    fkk = get_fkk_filter(prefer_gpu=True)

    # Config without AGC
    config_no_agc = FKKConfig(
        v_min=100.0, v_max=800.0,
        mode='reject',
        apply_agc=False,
        edge_method='pad_copy',
        pad_time_top_ms=50.0,
        pad_time_bottom_ms=50.0,
    )

    # Config with AGC (adaptive epsilon, no max_gain clipping)
    config_with_agc = FKKConfig(
        v_min=100.0, v_max=800.0,
        mode='reject',
        apply_agc=True,
        agc_window_ms=100.0,
        edge_method='pad_copy',
        pad_time_top_ms=50.0,
        pad_time_bottom_ms=50.0,
    )

    result_no_agc = fkk.apply_filter(vol, config_no_agc)
    result_with_agc = fkk.apply_filter(vol, config_with_agc)

    # Compare artifact levels at top
    orig_trace = vol.data[:, 10, 10]
    noagc_trace = result_no_agc.data[:, 10, 10]
    agc_trace = result_with_agc.data[:, 10, 10]

    diff_noagc = noagc_trace - orig_trace
    diff_agc = agc_trace - orig_trace

    print("Difference from original (top 30 samples):")
    print(f"  Without AGC - RMS: {np.sqrt((diff_noagc[:30]**2).mean()):.4f}")
    print(f"  With AGC    - RMS: {np.sqrt((diff_agc[:30]**2).mean()):.4f}")

    print("\nDifference from original (samples 40-60):")
    print(f"  Without AGC - RMS: {np.sqrt((diff_noagc[40:60]**2).mean()):.4f}")
    print(f"  With AGC    - RMS: {np.sqrt((diff_agc[40:60]**2).mean()):.4f}")

    ratio_noagc = np.sqrt((diff_noagc[:30]**2).mean()) / (np.sqrt((diff_noagc[40:60]**2).mean()) + 1e-10)
    ratio_agc = np.sqrt((diff_agc[:30]**2).mean()) / (np.sqrt((diff_agc[40:60]**2).mean()) + 1e-10)

    print(f"\nTop/Mid artifact ratio (without AGC): {ratio_noagc:.2f}")
    print(f"Top/Mid artifact ratio (with AGC): {ratio_agc:.2f}")

    if ratio_agc > ratio_noagc * 1.5:
        print("\n⚠️  CONFIRMED: AGC increases relative artifact level at top!")


def visualize_artifacts(save_path=None):
    """Create visualization of the artifact issue."""
    print("\n" + "=" * 60)
    print("Generating Visualization")
    print("=" * 60)

    vol = create_test_volume()

    try:
        from processors.fkk_filter_gpu import get_fkk_filter
        fkk = get_fkk_filter(prefer_gpu=True)
    except ImportError as e:
        print(f"Could not import FKK filter: {e}")
        return

    # Three scenarios - now all using adaptive epsilon (no max_gain clipping)
    configs = {
        'No AGC': FKKConfig(v_min=100, v_max=800, mode='reject', apply_agc=False),
        'AGC (window=100ms)': FKKConfig(v_min=100, v_max=800, mode='reject',
                                        apply_agc=True, agc_window_ms=100),
        'AGC (window=200ms)': FKKConfig(v_min=100, v_max=800, mode='reject',
                                       apply_agc=True, agc_window_ms=200),
    }

    results = {}
    for name, cfg in configs.items():
        results[name] = fkk.apply_filter(vol, cfg)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Original
    axes[0, 0].imshow(vol.data[:, :, 10].T, aspect='auto', cmap='seismic',
                       vmin=-2, vmax=2)
    axes[0, 0].set_title('Original')
    axes[0, 0].set_ylabel('Inline')

    # Results
    for i, (name, result) in enumerate(results.items()):
        axes[0, i+1].imshow(result.data[:, :, 10].T, aspect='auto', cmap='seismic',
                            vmin=-2, vmax=2)
        axes[0, i+1].set_title(f'FKK: {name}')

    # Differences
    axes[1, 0].set_visible(False)
    for i, (name, result) in enumerate(results.items()):
        diff = result.data - vol.data
        axes[1, i+1].imshow(diff[:, :, 10].T, aspect='auto', cmap='seismic',
                            vmin=-0.5, vmax=0.5)
        axes[1, i+1].set_title(f'Difference: {name}')
        axes[1, i+1].set_xlabel('Time sample')
        axes[1, i+1].set_ylabel('Inline')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


def run_all_tests():
    """Run all diagnostic tests."""
    test_agc_scale_factors_at_top()
    test_agc_reversal_amplifies_artifacts()
    test_temporal_taper_placement()
    test_pad_copy_temporal_continuity()
    test_full_fkk_with_agc_isolation()

    print("\n" + "=" * 60)
    print("SUMMARY: AGC Implementation with Adaptive Epsilon")
    print("=" * 60)
    print("""
IMPLEMENTED FIX: Removed max_gain clipping, using adaptive epsilon instead.

The max_gain parameter was causing 'shadow' artifacts because:
   - When outliers (air blasts) exit the AGC window, adjacent quiet zones
     need high gain to compensate
   - Clipping to max_gain created insufficient amplification
   - AGC reversal then created artifacts mirroring the outlier shape

NEW APPROACH (adaptive epsilon):
   - epsilon = 0.1% of global RMS (automatically computed)
   - Effective max_gain ≈ 1000 (natural limit from epsilon)
   - No clipping means perfect AGC reversal: (data * sf) / sf = data
   - Shadow artifacts eliminated

REMAINING CONSIDERATIONS:
   - Temporal pad_copy handles high-amplitude edges smoothly
   - AGC window size affects artifact suppression (longer = smoother)
   - For extreme outliers, consider muting before AGC
""")


if __name__ == '__main__':
    run_all_tests()
    # visualize_artifacts('/tmp/fkk_artifact_analysis.png')
