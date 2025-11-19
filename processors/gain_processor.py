"""
Gain processor - multiplies traces by a constant factor.

Simple processor used for testing and basic amplitude adjustment.
"""

import numpy as np
import sys
from processors.base_processor import BaseProcessor
from models.seismic_data import SeismicData


class GainProcessor(BaseProcessor):
    """
    Applies constant gain (multiplication) to all traces.

    Simple processor that multiplies every sample by a constant factor.
    Useful for amplitude scaling and testing.
    """

    def _validate_params(self):
        """Validate gain parameters."""
        if 'gain' not in self.params:
            raise ValueError("Missing required parameter: gain")

        self.gain = float(self.params['gain'])

        if self.gain == 0:
            raise ValueError("Gain cannot be zero")

    def process(self, data: SeismicData) -> SeismicData:
        """
        Apply constant gain to all traces.

        Args:
            data: Input seismic data

        Returns:
            Gain-adjusted seismic data
        """
        # Apply gain
        gained_traces = data.traces * self.gain

        # Create new SeismicData with processing metadata
        metadata = data.metadata.copy()
        if 'processing_history' not in metadata:
            metadata['processing_history'] = []
        metadata['processing_history'].append(self.get_description())

        return SeismicData(
            traces=gained_traces,
            sample_rate=data.sample_rate,
            headers=data.headers,
            metadata=metadata
        )

    def get_description(self) -> str:
        """Get description of this processor."""
        return f"Gain: {self.gain}x"
