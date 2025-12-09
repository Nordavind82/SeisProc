"""
Diagnostic test for frequency band filtering in TFDenoise STFT.

This test verifies that frequency parameters are correctly applied
and only the specified frequency band is processed.

IMPORTANT: SeismicData expects sample_rate in MILLISECONDS (seismic industry standard).
- sample_rate=2.0 means 2ms sampling interval = 500 Hz
- nyquist_freq = 1000 / (2 * sample_rate_ms) = 250 Hz for 2ms sampling
"""

import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

from processors.tf_denoise import TFDenoise
from models.seismic_data import SeismicData


def test_frequency_band_filtering():
    """
    Test that ONLY frequencies within [fmin, fmax] are processed.

    Strategy:
    1. Create signal with distinct frequency content at 20Hz, 100Hz, 200Hz
    2. Add different noise levels at each frequency
    3. Apply TFDenoise with fmin=80, fmax=249
    4. Verify: 20Hz should be UNCHANGED, 100Hz and 200Hz may change
    """
    logger.info("=" * 70)
    logger.info("DIAGNOSTIC: Frequency Band Filtering Test")
    logger.info("=" * 70)

    # Parameters - SeismicData expects sample_rate in MILLISECONDS
    sample_interval_ms = 2.0  # milliseconds (2ms = 500 Hz sampling)
    sample_rate_hz = 1000.0 / sample_interval_ms  # 500 Hz
    nyquist = sample_rate_hz / 2  # 250 Hz
    duration = 2.0  # seconds
    n_samples = int(sample_rate_hz * duration)
    n_traces = 20

    fmin = 80.0
    fmax = 249.0

    logger.info(f"\nTest configuration:")
    logger.info(f"  sample_interval = {sample_interval_ms} ms")
    logger.info(f"  sample_rate = {sample_rate_hz} Hz")
    logger.info(f"  nyquist = {nyquist} Hz")
    logger.info(f"  fmin = {fmin} Hz")
    logger.info(f"  fmax = {fmax} Hz")
    logger.info(f"  n_traces = {n_traces}")
    logger.info(f"  n_samples = {n_samples}")

    # Create time vector (in seconds for sine generation)
    t = np.arange(n_samples) / sample_rate_hz

    # Create signal with 3 distinct frequencies
    # 20 Hz - OUTSIDE band (below fmin)
    # 100 Hz - INSIDE band
    # 200 Hz - INSIDE band
    traces = np.zeros((n_samples, n_traces))

    for i in range(n_traces):
        # Clean signal components
        signal_20hz = 1.0 * np.sin(2 * np.pi * 20 * t)
        signal_100hz = 1.0 * np.sin(2 * np.pi * 100 * t)
        signal_200hz = 1.0 * np.sin(2 * np.pi * 200 * t)

        traces[:, i] = signal_20hz + signal_100hz + signal_200hz

    # Add spatially varying noise to trigger thresholding
    # Make one trace an "outlier" with extra noise
    np.random.seed(42)
    for i in range(n_traces):
        noise_level = 0.1 if i != n_traces // 2 else 0.5  # Middle trace is noisy
        traces[:, i] += np.random.randn(n_samples) * noise_level

    # Store original for comparison
    original_traces = traces.copy()

    # Create processor
    processor = TFDenoise(
        aperture=7,
        fmin=fmin,
        fmax=fmax,
        threshold_k=3.0,
        threshold_mode='adaptive',
        transform_type='stft'
    )

    logger.info(f"\nProcessor configuration:")
    logger.info(f"  transform_type: {processor.transform_type}")
    logger.info(f"  fmin: {processor.fmin} Hz")
    logger.info(f"  fmax: {processor.fmax} Hz")
    logger.info(f"  threshold_k: {processor.threshold_k}")
    logger.info(f"  threshold_mode: {processor.threshold_mode}")

    # Process - pass sample_rate in MILLISECONDS
    data = SeismicData(traces=traces, sample_rate=sample_interval_ms)
    logger.info(f"\nSeismicData check:")
    logger.info(f"  sample_rate (from data) = {data.sample_rate} ms")
    logger.info(f"  nyquist_freq (from data) = {data.nyquist_freq} Hz")
    result = processor.process(data)

    # Compute difference
    difference = original_traces - result.traces

    # Analyze spectrum of input, output, and difference
    def compute_band_energy(signal, freq_low, freq_high):
        """Compute energy in a frequency band."""
        spectrum = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(len(signal), 1/sample_rate_hz)
        mask = (freqs >= freq_low) & (freqs <= freq_high)
        return np.sum(spectrum[mask]**2)

    # Analyze middle trace (the noisy one)
    trace_idx = n_traces // 2

    input_trace = original_traces[:, trace_idx]
    output_trace = result.traces[:, trace_idx]
    diff_trace = difference[:, trace_idx]

    logger.info(f"\nEnergy analysis for trace {trace_idx} (the noisy outlier):")

    # Energy in different bands for INPUT
    input_20hz = compute_band_energy(input_trace, 15, 25)
    input_100hz = compute_band_energy(input_trace, 90, 110)
    input_200hz = compute_band_energy(input_trace, 190, 210)

    logger.info(f"\n  INPUT energy:")
    logger.info(f"    20Hz band (15-25Hz):   {input_20hz:.2f}")
    logger.info(f"    100Hz band (90-110Hz): {input_100hz:.2f}")
    logger.info(f"    200Hz band (190-210Hz): {input_200hz:.2f}")

    # Energy in different bands for OUTPUT
    output_20hz = compute_band_energy(output_trace, 15, 25)
    output_100hz = compute_band_energy(output_trace, 90, 110)
    output_200hz = compute_band_energy(output_trace, 190, 210)

    logger.info(f"\n  OUTPUT energy:")
    logger.info(f"    20Hz band (15-25Hz):   {output_20hz:.2f}")
    logger.info(f"    100Hz band (90-110Hz): {output_100hz:.2f}")
    logger.info(f"    200Hz band (190-210Hz): {output_200hz:.2f}")

    # Energy in different bands for DIFFERENCE
    diff_20hz = compute_band_energy(diff_trace, 15, 25)
    diff_100hz = compute_band_energy(diff_trace, 90, 110)
    diff_200hz = compute_band_energy(diff_trace, 190, 210)
    diff_below_fmin = compute_band_energy(diff_trace, 0, fmin)
    diff_in_band = compute_band_energy(diff_trace, fmin, fmax)

    logger.info(f"\n  DIFFERENCE energy (what was removed):")
    logger.info(f"    20Hz band (15-25Hz):   {diff_20hz:.2f}")
    logger.info(f"    100Hz band (90-110Hz): {diff_100hz:.2f}")
    logger.info(f"    200Hz band (190-210Hz): {diff_200hz:.2f}")
    logger.info(f"    Total below fmin (0-{fmin}Hz): {diff_below_fmin:.2f}")
    logger.info(f"    Total in band ({fmin}-{fmax}Hz): {diff_in_band:.2f}")

    # Check if 20Hz is preserved
    change_20hz = abs(input_20hz - output_20hz) / input_20hz if input_20hz > 0 else 0

    logger.info(f"\n  VERDICT:")
    logger.info(f"    20Hz change: {change_20hz*100:.2f}%")

    if change_20hz < 0.01:  # Less than 1% change
        logger.info(f"    ✓ 20Hz (outside band) is PRESERVED as expected")
    else:
        logger.warning(f"    ✗ 20Hz (outside band) was MODIFIED - this is a BUG!")

    if diff_below_fmin < 1e-6:
        logger.info(f"    ✓ No energy removed below fmin")
    else:
        logger.warning(f"    ✗ Energy was removed below fmin - frequency filtering not working!")

    # Full spectrum analysis
    logger.info(f"\n  Full spectrum comparison:")
    input_spectrum = np.abs(np.fft.rfft(input_trace))
    output_spectrum = np.abs(np.fft.rfft(output_trace))
    diff_spectrum = np.abs(np.fft.rfft(diff_trace))
    freqs = np.fft.rfftfreq(n_samples, 1/sample_rate_hz)

    # Find where difference is significant
    significant_diff = diff_spectrum > np.max(diff_spectrum) * 0.1
    if np.any(significant_diff):
        sig_freqs = freqs[significant_diff]
        logger.info(f"    Significant difference at frequencies: {sig_freqs.min():.1f} - {sig_freqs.max():.1f} Hz")

        # Check if any significant difference is outside the band
        outside_band = (sig_freqs < fmin) | (sig_freqs > fmax)
        if np.any(outside_band):
            logger.warning(f"    ✗ PROBLEM: Difference found outside [{fmin}, {fmax}] Hz band!")
            logger.warning(f"       Frequencies with difference outside band: {sig_freqs[outside_band]}")
            return False
        else:
            logger.info(f"    ✓ All significant differences are within [{fmin}, {fmax}] Hz band")
    else:
        logger.info(f"    No significant differences detected (minimal denoising)")

    return True


def test_why_all_frequencies_in_difference():
    """
    Investigate why all frequencies appear in the difference.

    Hypothesis: The issue is NOT with frequency filtering, but with
    the fact that modifying ANY frequency in STFT affects ALL frequencies
    in the reconstructed signal due to windowing/overlap-add effects.
    """
    logger.info("\n" + "=" * 70)
    logger.info("DIAGNOSTIC: Why all frequencies appear in difference")
    logger.info("=" * 70)

    sample_rate = 500.0
    n_samples = 1000
    n_traces = 10

    t = np.arange(n_samples) / sample_rate

    # Create signal with only LOW frequency (20 Hz)
    traces = np.zeros((n_samples, n_traces))
    for i in range(n_traces):
        traces[:, i] = np.sin(2 * np.pi * 20 * t)

    # Add noise to trigger thresholding
    np.random.seed(42)
    traces += np.random.randn(n_samples, n_traces) * 0.1

    original = traces.copy()

    # Process with fmin=80 (so 20Hz should NOT be touched)
    processor = TFDenoise(
        aperture=5,
        fmin=80.0,
        fmax=240.0,
        threshold_k=3.0,
        threshold_mode='adaptive',
        transform_type='stft'
    )

    data = SeismicData(traces=traces, sample_rate=sample_rate)
    result = processor.process(data)

    diff = original - result.traces

    # Check total difference
    total_diff_energy = np.sum(diff**2)
    total_input_energy = np.sum(original**2)

    logger.info(f"\nTest: Signal has only 20Hz, processing band is 80-240Hz")
    logger.info(f"  Total input energy: {total_input_energy:.2f}")
    logger.info(f"  Total difference energy: {total_diff_energy:.6f}")
    logger.info(f"  Difference ratio: {total_diff_energy/total_input_energy*100:.4f}%")

    if total_diff_energy < 1e-10:
        logger.info(f"  ✓ No change detected - frequency filtering works correctly!")
        return True
    else:
        logger.warning(f"  ✗ Change detected even though signal is outside processing band!")

        # Analyze where the difference comes from
        diff_spectrum = np.abs(np.fft.rfft(diff[:, 0]))
        freqs = np.fft.rfftfreq(n_samples, 1/sample_rate)

        peak_idx = np.argmax(diff_spectrum)
        logger.info(f"  Peak difference at: {freqs[peak_idx]:.1f} Hz")

        return False


if __name__ == '__main__':
    test_frequency_band_filtering()
    test_why_all_frequencies_in_difference()
