"""
Bandpass filter processor - zero-phase Butterworth bandpass filter.
Properly handles Nyquist frequency and applies zero-phase filtering.
"""
import numpy as np
from scipy import signal
import sys
from processors.base_processor import BaseProcessor
from models.seismic_data import SeismicData


class BandpassFilter(BaseProcessor):
    """
    Zero-phase Butterworth bandpass filter.

    Applies forward-backward filtering to achieve zero phase shift,
    which is critical for seismic QC to avoid time shifts.
    """

    def _validate_params(self):
        """Validate bandpass parameters."""
        required = ['low_freq', 'high_freq']
        for param in required:
            if param not in self.params:
                raise ValueError(f"Missing required parameter: {param}")

        self.low_freq = float(self.params['low_freq'])
        self.high_freq = float(self.params['high_freq'])
        self.order = int(self.params.get('order', 4))  # Default 4th order

        if self.low_freq <= 0:
            raise ValueError(f"Low frequency must be positive, got {self.low_freq}")

        if self.high_freq <= self.low_freq:
            raise ValueError(
                f"High frequency ({self.high_freq}) must be > low frequency ({self.low_freq})"
            )

        if self.order < 1 or self.order > 10:
            raise ValueError(f"Filter order must be in [1, 10], got {self.order}")

    def process(self, data: SeismicData) -> SeismicData:
        """
        Apply zero-phase bandpass filter.

        Args:
            data: Input seismic data

        Returns:
            Filtered seismic data

        Raises:
            ValueError: If filter frequencies exceed Nyquist
        """
        # Validate against Nyquist frequency
        nyquist = data.nyquist_freq
        if self.high_freq >= nyquist:
            raise ValueError(
                f"High frequency ({self.high_freq} Hz) must be < Nyquist frequency "
                f"({nyquist:.1f} Hz). Reduce frequency or check sample rate."
            )

        # Design Butterworth bandpass filter
        # Normalized frequencies (relative to Nyquist)
        low_norm = self.low_freq / nyquist
        high_norm = self.high_freq / nyquist

        # Create filter coefficients
        sos = signal.butter(
            self.order,
            [low_norm, high_norm],
            btype='bandpass',
            output='sos'
        )

        # Apply zero-phase filtering to each trace
        filtered_traces = np.zeros_like(data.traces)
        for i in range(data.n_traces):
            # sosfiltfilt applies forward-backward filtering (zero phase)
            filtered_traces[:, i] = signal.sosfiltfilt(sos, data.traces[:, i])

        # Create new SeismicData with processing metadata
        metadata = data.metadata.copy()
        if 'processing_history' not in metadata:
            metadata['processing_history'] = []
        metadata['processing_history'].append(self.get_description())

        return SeismicData(
            traces=filtered_traces,
            sample_rate=data.sample_rate,
            headers=data.headers,
            metadata=metadata
        )

    def get_description(self) -> str:
        """Get description of this filter."""
        return (f"Zero-phase Butterworth bandpass: {self.low_freq}-{self.high_freq} Hz, "
                f"order {self.order}")


class ProcessingPipeline:
    """
    Chain of processors applied sequentially.
    Makes it easy to add multiple processing steps.
    """

    def __init__(self):
        self.processors = []

    def add_processor(self, processor: BaseProcessor):
        """Add a processor to the pipeline."""
        self.processors.append(processor)

    def clear(self):
        """Remove all processors."""
        self.processors.clear()

    def process(self, data: SeismicData) -> SeismicData:
        """
        Apply all processors in sequence.

        Args:
            data: Input seismic data

        Returns:
            Processed data after all processors
        """
        result = data
        for processor in self.processors:
            result = processor.process(result)
        return result

    def get_description(self) -> str:
        """Get description of entire pipeline."""
        if not self.processors:
            return "No processing applied"

        steps = [f"{i+1}. {proc.get_description()}"
                 for i, proc in enumerate(self.processors)]
        return "Processing pipeline:\n" + "\n".join(steps)

    def __repr__(self) -> str:
        return f"ProcessingPipeline(steps={len(self.processors)})"
