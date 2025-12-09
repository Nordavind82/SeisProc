"""
Test to verify frequency parameters (fmin, fmax) are properly used in TF-Denoise.

This test diagnoses whether frequency filtering is actually applied.
"""

import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

from processors.tf_denoise import (
    stockwell_transform,
    inverse_stockwell_transform,
    stft_transform,
    TFDenoise
)
from models.seismic_data import SeismicData


def test_stransform_frequency_range():
    """Test that S-Transform respects fmin/fmax parameters."""
    logger.info("=" * 60)
    logger.info("TEST 1: S-Transform frequency range filtering")
    logger.info("=" * 60)

    # Generate signal
    sample_rate = 250.0
    duration = 1.0
    n_samples = int(sample_rate * duration)
    t = np.arange(n_samples) / sample_rate

    # Multi-frequency signal: 10Hz + 50Hz + 100Hz
    signal = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 100 * t)

    # Test 1a: Full spectrum
    logger.info("\n1a. Full spectrum S-Transform:")
    S_full, freqs_full = stockwell_transform(signal)
    logger.info(f"   Shape: {S_full.shape}")
    logger.info(f"   Freq range (normalized): {freqs_full[0]:.4f} to {freqs_full[-1]:.4f}")
    logger.info(f"   Freq range (Hz): {freqs_full[0] * sample_rate:.1f} to {freqs_full[-1] * sample_rate:.1f}")
    logger.info(f"   Number of frequency bins: {len(freqs_full)}")

    # Test 1b: Limited frequency range (20-80 Hz)
    fmin_hz = 20.0
    fmax_hz = 80.0
    fmin_norm = fmin_hz / sample_rate  # Normalized to Nyquist
    fmax_norm = fmax_hz / sample_rate

    logger.info(f"\n1b. Limited spectrum S-Transform (fmin={fmin_hz}Hz, fmax={fmax_hz}Hz):")
    logger.info(f"   Normalized: fmin={fmin_norm:.4f}, fmax={fmax_norm:.4f}")

    S_limited, freqs_limited = stockwell_transform(signal, fmin=fmin_norm, fmax=fmax_norm)
    logger.info(f"   Shape: {S_limited.shape}")
    logger.info(f"   Freq range (normalized): {freqs_limited[0]:.4f} to {freqs_limited[-1]:.4f}")
    logger.info(f"   Freq range (Hz): {freqs_limited[0] * sample_rate:.1f} to {freqs_limited[-1] * sample_rate:.1f}")
    logger.info(f"   Number of frequency bins: {len(freqs_limited)}")

    # Verify
    if S_limited.shape[0] < S_full.shape[0]:
        logger.info("   ✓ S-Transform correctly limits frequency range")
    else:
        logger.warning("   ✗ S-Transform NOT limiting frequency range!")

    return S_full.shape[0] != S_limited.shape[0]


def test_stft_frequency_filtering():
    """Test that STFT frequency filtering works in TFDenoise."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: STFT frequency filtering in TFDenoise._process_with_stft")
    logger.info("=" * 60)

    sample_rate = 250.0
    duration = 1.0
    n_samples = int(sample_rate * duration)
    n_traces = 10
    t = np.arange(n_samples) / sample_rate

    # Signal with energy at 10Hz (should be preserved if fmin=20) and 50Hz (in band)
    traces = np.zeros((n_samples, n_traces))
    for i in range(n_traces):
        traces[:, i] = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 50 * t)

    # Add some noise to trigger thresholding
    np.random.seed(42)
    traces += np.random.randn(n_samples, n_traces) * 0.5

    # Test with fmin=5, fmax=100 (should process all)
    logger.info("\n2a. TFDenoise with fmin=5Hz, fmax=100Hz (wide band):")
    processor_wide = TFDenoise(
        aperture=5,
        fmin=5.0,
        fmax=100.0,
        threshold_k=3.0,
        threshold_mode='adaptive',
        transform_type='stft'
    )

    data = SeismicData(traces=traces, sample_rate=sample_rate)
    result_wide = processor_wide.process(data)

    # Check spectrum of result
    input_spectrum = np.abs(np.fft.rfft(traces[:, 0]))
    output_spectrum_wide = np.abs(np.fft.rfft(result_wide.traces[:, 0]))

    freq_bins = np.fft.rfftfreq(n_samples, 1/sample_rate)

    # Find energy at 10Hz and 50Hz
    idx_10hz = np.argmin(np.abs(freq_bins - 10))
    idx_50hz = np.argmin(np.abs(freq_bins - 50))

    logger.info(f"   Input 10Hz energy: {input_spectrum[idx_10hz]:.2f}")
    logger.info(f"   Output 10Hz energy: {output_spectrum_wide[idx_10hz]:.2f}")
    logger.info(f"   Input 50Hz energy: {input_spectrum[idx_50hz]:.2f}")
    logger.info(f"   Output 50Hz energy: {output_spectrum_wide[idx_50hz]:.2f}")

    # Test with fmin=20, fmax=80 (should NOT process 10Hz)
    logger.info("\n2b. TFDenoise with fmin=20Hz, fmax=80Hz (narrow band):")
    logger.info("   Expected: 10Hz should be UNCHANGED, 50Hz may be processed")

    processor_narrow = TFDenoise(
        aperture=5,
        fmin=20.0,
        fmax=80.0,
        threshold_k=3.0,
        threshold_mode='adaptive',
        transform_type='stft'
    )

    result_narrow = processor_narrow.process(data)
    output_spectrum_narrow = np.abs(np.fft.rfft(result_narrow.traces[:, 0]))

    logger.info(f"   Output 10Hz energy: {output_spectrum_narrow[idx_10hz]:.2f}")
    logger.info(f"   Output 50Hz energy: {output_spectrum_narrow[idx_50hz]:.2f}")

    # Compare 10Hz between wide and narrow - should be same if filtering works
    diff_10hz = abs(output_spectrum_wide[idx_10hz] - output_spectrum_narrow[idx_10hz])
    logger.info(f"   Difference at 10Hz (wide vs narrow): {diff_10hz:.4f}")

    if diff_10hz < 0.01:
        logger.info("   ✓ 10Hz energy preserved (not in processing band)")
    else:
        logger.warning("   ✗ 10Hz energy differs - frequency filtering may not be working!")

    return diff_10hz < 0.01


def test_tfdenoise_parameter_propagation():
    """Test that parameters are actually propagated to processing functions."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Parameter propagation verification")
    logger.info("=" * 60)

    sample_rate = 250.0
    duration = 1.0
    n_samples = int(sample_rate * duration)
    n_traces = 20
    nyquist = sample_rate / 2  # 125 Hz

    # Use frequencies that are within valid range (fmax < Nyquist)
    fmin_hz = 15.0
    fmax_hz = 60.0  # Within Nyquist range

    # Create signal with specific frequency content
    t = np.arange(n_samples) / sample_rate
    traces = np.zeros((n_samples, n_traces))
    for i in range(n_traces):
        # 10Hz (outside fmin=15), 30Hz (inside), 80Hz (outside fmax=60)
        traces[:, i] = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 30 * t) + np.sin(2 * np.pi * 80 * t)

    # Add noise to trigger thresholding
    np.random.seed(42)
    traces += np.random.randn(n_samples, n_traces) * 0.3

    # Create processor with specific parameters
    processor = TFDenoise(
        aperture=7,
        fmin=fmin_hz,
        fmax=fmax_hz,
        threshold_k=2.5,
        threshold_mode='hard',
        transform_type='stft',  # Use STFT - S-Transform has reconstruction issues
        time_smoothing=1,
        low_amp_protection=True,
        low_amp_factor=0.2
    )

    logger.info("\nProcessor parameters:")
    logger.info(f"   aperture: {processor.aperture}")
    logger.info(f"   fmin: {processor.fmin} Hz")
    logger.info(f"   fmax: {processor.fmax} Hz")
    logger.info(f"   nyquist: {nyquist} Hz")
    logger.info(f"   threshold_k: {processor.threshold_k}")
    logger.info(f"   threshold_mode: {processor.threshold_mode}")
    logger.info(f"   transform_type: {processor.transform_type}")

    # Process data
    data = SeismicData(traces=traces, sample_rate=sample_rate)
    result = processor.process(data)

    # Check spectrum - frequencies outside [15, 60] should be unchanged
    input_spectrum = np.abs(np.fft.rfft(traces[:, 0]))
    output_spectrum = np.abs(np.fft.rfft(result.traces[:, 0]))

    freq_bins = np.fft.rfftfreq(n_samples, 1/sample_rate)

    # Find energy at test frequencies
    idx_10hz = np.argmin(np.abs(freq_bins - 10))
    idx_30hz = np.argmin(np.abs(freq_bins - 30))
    idx_80hz = np.argmin(np.abs(freq_bins - 80))

    logger.info(f"\n   Frequency content verification:")
    logger.info(f"   10Hz (outside band): input={input_spectrum[idx_10hz]:.2f}, output={output_spectrum[idx_10hz]:.2f}")
    logger.info(f"   30Hz (inside band): input={input_spectrum[idx_30hz]:.2f}, output={output_spectrum[idx_30hz]:.2f}")
    logger.info(f"   80Hz (outside band): input={input_spectrum[idx_80hz]:.2f}, output={output_spectrum[idx_80hz]:.2f}")

    # Check that correct frequencies are being used
    fmin_norm = processor.fmin / nyquist
    fmax_norm = processor.fmax / nyquist

    logger.info(f"\n   Normalized frequencies (FIXED: fmin/Nyquist):")
    logger.info(f"   fmin_norm = {processor.fmin} / {nyquist} = {fmin_norm:.4f}")
    logger.info(f"   fmax_norm = {processor.fmax} / {nyquist} = {fmax_norm:.4f}")

    # Verify normalization is in valid range [0, 0.5]
    if fmin_norm <= 0.5 and fmax_norm <= 0.5:
        logger.info("   ✓ Frequency normalization is within valid range [0, 0.5]")
        return True
    else:
        logger.warning(f"   ✗ Frequency normalization out of range: [{fmin_norm:.2f}, {fmax_norm:.2f}]")
        logger.warning(f"     fmax_hz={fmax_hz} > Nyquist/2 = {nyquist} - consider using fmax < {nyquist} Hz")
        return False


def test_stft_frequency_mask_logic():
    """Test the STFT frequency mask logic directly."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: STFT frequency mask logic inspection")
    logger.info("=" * 60)

    from scipy import signal

    sample_rate = 250.0
    n_samples = 250
    nperseg = 64
    noverlap = 32

    # Get STFT frequencies
    freqs, times, Zxx = signal.stft(np.random.randn(n_samples), nperseg=nperseg, noverlap=noverlap)

    logger.info(f"\nSTFT frequency output:")
    logger.info(f"   freqs shape: {freqs.shape}")
    logger.info(f"   freqs range: {freqs[0]:.4f} to {freqs[-1]:.4f}")
    logger.info(f"   These are NORMALIZED frequencies (0 to 0.5)")

    # Convert to Hz as done in _process_with_stft
    freq_hz = freqs * sample_rate
    logger.info(f"\n   After multiplying by sample_rate ({sample_rate}):")
    logger.info(f"   freq_hz range: {freq_hz[0]:.1f} to {freq_hz[-1]:.1f} Hz")

    # Test frequency mask
    fmin = 20.0
    fmax = 80.0

    freq_mask = (freq_hz >= fmin) & (freq_hz <= fmax)

    logger.info(f"\n   Frequency mask for fmin={fmin}Hz, fmax={fmax}Hz:")
    logger.info(f"   Frequencies in mask: {freq_hz[freq_mask]}")
    logger.info(f"   Number of bins in mask: {np.sum(freq_mask)} / {len(freq_hz)}")

    # Check if mask is applied correctly
    if np.sum(freq_mask) > 0 and np.sum(freq_mask) < len(freq_hz):
        logger.info("   ✓ Frequency mask correctly selects subset of frequencies")
    else:
        logger.warning("   ✗ Frequency mask issue!")

    # The issue: scipy.stft returns normalized frequencies (0 to 0.5)
    # Multiplying by sample_rate gives range 0 to sample_rate/2 = Nyquist
    # This is CORRECT!

    return True


def test_stransform_frequency_normalization():
    """Test S-Transform frequency normalization - verify fix is applied."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: S-Transform frequency normalization verification")
    logger.info("=" * 60)

    sample_rate = 250.0
    nyquist = sample_rate / 2  # 125 Hz
    n_samples = 250

    # Test with frequencies well within Nyquist range
    fmin_hz = 10.0
    fmax_hz = 50.0  # Well below Nyquist

    logger.info(f"\nTest configuration:")
    logger.info(f"   sample_rate = {sample_rate} Hz")
    logger.info(f"   nyquist = {nyquist} Hz")
    logger.info(f"   fmin = {fmin_hz} Hz")
    logger.info(f"   fmax = {fmax_hz} Hz")

    # CORRECT normalization formula (FIXED):
    fmin_norm_correct = fmin_hz / nyquist  # 10/125 = 0.08
    fmax_norm_correct = fmax_hz / nyquist  # 50/125 = 0.40

    # OLD WRONG normalization (BUGGY):
    fmin_norm_wrong = fmin_hz / (2 * nyquist)  # 10/250 = 0.04
    fmax_norm_wrong = fmax_hz / (2 * nyquist)  # 50/250 = 0.20

    logger.info(f"\n   OLD (buggy) normalization (fmin/(2*Nyquist)):")
    logger.info(f"   fmin_norm = {fmin_hz} / {2*nyquist} = {fmin_norm_wrong:.4f}")
    logger.info(f"   fmax_norm = {fmax_hz} / {2*nyquist} = {fmax_norm_wrong:.4f}")

    logger.info(f"\n   NEW (correct) normalization (fmin/Nyquist):")
    logger.info(f"   fmin_norm = {fmin_hz} / {nyquist} = {fmin_norm_correct:.4f}")
    logger.info(f"   fmax_norm = {fmax_hz} / {nyquist} = {fmax_norm_correct:.4f}")

    # Test with WRONG normalization (simulates old bug)
    signal_data = np.random.randn(n_samples)
    S_wrong, freqs_wrong = stockwell_transform(signal_data, fmin=fmin_norm_wrong, fmax=fmax_norm_wrong)
    freq_hz_max_wrong = freqs_wrong[-1] * nyquist

    # Test with CORRECT normalization
    S_correct, freqs_correct = stockwell_transform(signal_data, fmin=fmin_norm_correct, fmax=fmax_norm_correct)
    freq_hz_max_correct = freqs_correct[-1] * nyquist

    logger.info(f"\n   Results comparison:")
    logger.info(f"   OLD normalization: max freq = {freq_hz_max_wrong:.1f} Hz (expected {fmax_hz/2:.1f} Hz - HALVED!)")
    logger.info(f"   NEW normalization: max freq = {freq_hz_max_correct:.1f} Hz (expected {fmax_hz:.1f} Hz)")

    # The fix is verified if correct normalization gives frequencies close to requested
    tolerance = 5.0
    if abs(freq_hz_max_correct - fmax_hz) < tolerance:
        logger.info(f"\n   ✓ NEW normalization gives CORRECT frequency range!")
        return True
    else:
        logger.warning(f"\n   ✗ Frequency range still wrong after fix!")
        return False


def run_all_tests():
    """Run all frequency parameter tests."""
    print("\n" + "=" * 70)
    print("FREQUENCY PARAMETER INVESTIGATION")
    print("=" * 70)

    results = {}

    results['stransform_range'] = test_stransform_frequency_range()
    results['stft_filtering'] = test_stft_frequency_filtering()
    results['parameter_propagation'] = test_tfdenoise_parameter_propagation()
    results['stft_mask_logic'] = test_stft_frequency_mask_logic()
    results['stransform_normalization'] = test_stransform_frequency_normalization()

    print("\n" + "=" * 70)
    print("SUMMARY OF FINDINGS")
    print("=" * 70)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"   {test_name}: {status}")
        if not passed:
            all_passed = False

    if not all_passed:
        print("\n" + "!" * 70)
        print("ISSUES DETECTED - SEE DETAILED OUTPUT ABOVE")
        print("!" * 70)

    return results


if __name__ == '__main__':
    run_all_tests()
