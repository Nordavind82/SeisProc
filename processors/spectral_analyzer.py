"""
Spectral analyzer - computes frequency spectrum from seismic traces.
Simple FFT-based analysis for seismic QC.
"""
import numpy as np
from scipy.fft import rfft, rfftfreq
from typing import Tuple


class SpectralAnalyzer:
    """
    Computes amplitude spectrum from seismic trace data.

    Simple FFT-based analysis suitable for seismic QC:
    - Frequency range: 0 to Nyquist
    - Amplitude in dB scale
    - Single trace or ensemble average
    """

    def __init__(self, sample_rate_ms: float):
        """
        Initialize spectral analyzer.

        Args:
            sample_rate_ms: Sample rate in milliseconds
        """
        self.sample_rate_ms = sample_rate_ms
        self.sample_rate_hz = 1000.0 / sample_rate_ms

    def compute_spectrum(self, trace: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute amplitude spectrum for a single trace.

        Args:
            trace: Single trace data (1D array)

        Returns:
            Tuple of (frequencies, amplitudes_db)
            - frequencies: Array of frequency values in Hz
            - amplitudes_db: Amplitude spectrum in dB
        """
        # Compute FFT (real FFT for efficiency)
        spectrum = rfft(trace)

        # Compute frequencies
        n_samples = len(trace)
        frequencies = rfftfreq(n_samples, d=self.sample_rate_ms / 1000.0)

        # Convert to magnitude (amplitude spectrum)
        amplitudes = np.abs(spectrum)

        # Convert to dB scale (avoid log of zero)
        amplitudes_db = 20 * np.log10(amplitudes + 1e-10)

        return frequencies, amplitudes_db

    def compute_average_spectrum(self, traces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute average amplitude spectrum across multiple traces.

        Args:
            traces: Trace data (n_samples, n_traces)

        Returns:
            Tuple of (frequencies, avg_amplitudes_db)
        """
        n_samples, n_traces = traces.shape

        # Compute spectrum for first trace to get frequency array
        frequencies = rfftfreq(n_samples, d=self.sample_rate_ms / 1000.0)

        # Accumulate amplitudes
        sum_amplitudes = np.zeros(len(frequencies))

        for i in range(n_traces):
            spectrum = rfft(traces[:, i])
            sum_amplitudes += np.abs(spectrum)

        # Average
        avg_amplitudes = sum_amplitudes / n_traces

        # Convert to dB
        avg_amplitudes_db = 20 * np.log10(avg_amplitudes + 1e-10)

        return frequencies, avg_amplitudes_db

    def compute_spectrum_with_phase(self, trace: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute amplitude and phase spectrum for a single trace.

        Args:
            trace: Single trace data (1D array)

        Returns:
            Tuple of (frequencies, amplitudes_db, phase_degrees)
            - frequencies: Array of frequency values in Hz
            - amplitudes_db: Amplitude spectrum in dB
            - phase_degrees: Phase spectrum in degrees (-180 to 180)
        """
        # Compute FFT (real FFT for efficiency)
        spectrum = rfft(trace)

        # Compute frequencies
        n_samples = len(trace)
        frequencies = rfftfreq(n_samples, d=self.sample_rate_ms / 1000.0)

        # Amplitude spectrum
        amplitudes = np.abs(spectrum)
        amplitudes_db = 20 * np.log10(amplitudes + 1e-10)

        # Phase spectrum (in degrees)
        phase_radians = np.angle(spectrum)
        phase_degrees = np.degrees(phase_radians)

        return frequencies, amplitudes_db, phase_degrees

    def compute_average_spectrum_with_phase(self, traces: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute average amplitude and phase spectrum across multiple traces.

        Args:
            traces: Trace data (n_samples, n_traces)

        Returns:
            Tuple of (frequencies, avg_amplitudes_db, avg_phase_degrees)
        """
        n_samples, n_traces = traces.shape

        # Compute spectrum for first trace to get frequency array
        frequencies = rfftfreq(n_samples, d=self.sample_rate_ms / 1000.0)

        # Accumulate complex spectra for proper phase averaging
        sum_spectrum = np.zeros(len(frequencies), dtype=complex)

        for i in range(n_traces):
            spectrum = rfft(traces[:, i])
            sum_spectrum += spectrum

        # Average complex spectrum
        avg_spectrum = sum_spectrum / n_traces

        # Amplitude (dB)
        avg_amplitudes = np.abs(avg_spectrum)
        avg_amplitudes_db = 20 * np.log10(avg_amplitudes + 1e-10)

        # Phase (degrees)
        avg_phase_degrees = np.degrees(np.angle(avg_spectrum))

        return frequencies, avg_amplitudes_db, avg_phase_degrees

    def find_dominant_frequency(self, frequencies: np.ndarray,
                               amplitudes_db: np.ndarray,
                               freq_range: Tuple[float, float] = None) -> float:
        """
        Find dominant (peak) frequency in spectrum.

        Args:
            frequencies: Frequency array
            amplitudes_db: Amplitude spectrum in dB
            freq_range: Optional (min_freq, max_freq) to search within

        Returns:
            Dominant frequency in Hz
        """
        # Apply frequency range filter if specified
        if freq_range is not None:
            min_freq, max_freq = freq_range
            mask = (frequencies >= min_freq) & (frequencies <= max_freq)
            search_freqs = frequencies[mask]
            search_amps = amplitudes_db[mask]
        else:
            search_freqs = frequencies
            search_amps = amplitudes_db

        # Find peak
        peak_idx = np.argmax(search_amps)
        return search_freqs[peak_idx]
