"""
Seismic data model - container for seismic traces and metadata.
Follows best practices: immutable data, clear domain separation.
"""
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class SeismicData:
    """
    Immutable container for seismic data.

    Attributes:
        traces: 2D array (n_samples, n_traces) of seismic amplitudes
        sample_rate: Sample rate in milliseconds (e.g., 2.0 for 2ms)
        headers: Dictionary of trace headers (optional)
        metadata: Additional metadata (survey info, processing history, etc.)
    """
    traces: np.ndarray
    sample_rate: float  # in milliseconds
    headers: Optional[Dict[str, np.ndarray]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate data integrity."""
        if self.traces.ndim != 2:
            raise ValueError(f"Traces must be 2D array, got shape {self.traces.shape}")

        if self.sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {self.sample_rate}")

        # Ensure traces are float for processing
        if not np.issubdtype(self.traces.dtype, np.floating):
            object.__setattr__(self, 'traces', self.traces.astype(np.float32))

    @property
    def n_samples(self) -> int:
        """Number of time samples."""
        return self.traces.shape[0]

    @property
    def n_traces(self) -> int:
        """Number of traces."""
        return self.traces.shape[1]

    @property
    def duration(self) -> float:
        """Total duration in milliseconds."""
        return (self.n_samples - 1) * self.sample_rate

    @property
    def nyquist_freq(self) -> float:
        """Nyquist frequency in Hz."""
        return 1000.0 / (2.0 * self.sample_rate)

    def get_time_axis(self) -> np.ndarray:
        """Get time axis in milliseconds."""
        return np.arange(self.n_samples) * self.sample_rate

    def get_trace_axis(self) -> np.ndarray:
        """Get trace axis (trace numbers)."""
        return np.arange(self.n_traces)

    @property
    def coordinate_units(self) -> str:
        """
        Get coordinate units from metadata.

        Returns:
            'meters' or 'feet' (defaults to 'meters' if not specified)
        """
        return self.metadata.get('coordinate_units', 'meters')

    @property
    def unit_symbol(self) -> str:
        """
        Get short unit symbol for display.

        Returns:
            'm' for meters, 'ft' for feet
        """
        return 'm' if self.coordinate_units == 'meters' else 'ft'

    def copy(self) -> 'SeismicData':
        """Create a deep copy of this data."""
        return SeismicData(
            traces=self.traces.copy(),
            sample_rate=self.sample_rate,
            headers=self.headers.copy() if self.headers else None,
            metadata=self.metadata.copy()
        )

    def __repr__(self) -> str:
        return (f"SeismicData(n_samples={self.n_samples}, n_traces={self.n_traces}, "
                f"sample_rate={self.sample_rate}ms, nyquist={self.nyquist_freq:.1f}Hz)")
