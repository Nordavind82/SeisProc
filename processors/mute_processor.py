"""
Mute processor for applying top/bottom mutes based on velocity and offset.

Linear mute formula: T = offset / velocity
Applies cosine taper for smooth transition at mute boundary.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class MuteConfig:
    """Configuration for mute application."""
    velocity: float           # Mute velocity in m/s
    top_mute: bool = False    # Apply top mute (zero samples before mute time)
    bottom_mute: bool = False # Apply bottom mute (zero samples after mute time)
    taper_samples: int = 20   # Number of samples for cosine taper transition

    def __post_init__(self):
        """Validate configuration."""
        if self.velocity <= 0:
            raise ValueError(f"Velocity must be positive, got {self.velocity}")
        if self.taper_samples < 0:
            raise ValueError(f"Taper samples must be non-negative, got {self.taper_samples}")
        if not self.top_mute and not self.bottom_mute:
            raise ValueError("At least one of top_mute or bottom_mute must be True")

    def to_dict(self):
        """Serialize to dictionary for worker transfer."""
        return {
            'velocity': self.velocity,
            'top_mute': self.top_mute,
            'bottom_mute': self.bottom_mute,
            'taper_samples': self.taper_samples
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'MuteConfig':
        """Deserialize from dictionary."""
        return cls(**d)


class MuteProcessor:
    """
    Applies linear mute based on offset and velocity.

    Mute time calculation:
        T_mute = |offset| / velocity

    Where:
        - offset is in meters (from trace header)
        - velocity is in m/s
        - T_mute is in seconds

    Top mute: zeros samples before T_mute
    Bottom mute: zeros samples after T_mute
    """

    def __init__(self, config: MuteConfig):
        """
        Initialize mute processor.

        Args:
            config: MuteConfig with velocity and mute type settings
        """
        self.config = config
        # Pre-compute cosine taper
        if config.taper_samples > 0:
            self._taper = 0.5 * (1 - np.cos(np.linspace(0, np.pi, config.taper_samples)))
        else:
            self._taper = np.array([])

    def calculate_mute_sample(
        self,
        offset: float,
        sample_interval_ms: float
    ) -> int:
        """
        Calculate mute sample index for a given offset.

        Args:
            offset: Offset in meters (absolute value used)
            sample_interval_ms: Sample interval in milliseconds

        Returns:
            Sample index where mute should be applied
        """
        # Convert velocity to m/ms for consistent units
        velocity_m_per_ms = self.config.velocity / 1000.0

        # Calculate mute time in milliseconds
        mute_time_ms = abs(offset) / velocity_m_per_ms

        # Convert to sample index
        return int(mute_time_ms / sample_interval_ms)

    def apply_mute(
        self,
        trace: np.ndarray,
        offset: float,
        sample_interval_ms: float
    ) -> np.ndarray:
        """
        Apply mute to a single trace.

        Args:
            trace: 1D trace array (n_samples,)
            offset: Offset in meters
            sample_interval_ms: Sample interval in milliseconds

        Returns:
            Muted trace (copy, input unchanged)
        """
        result = trace.copy()
        n_samples = len(trace)

        mute_sample = self.calculate_mute_sample(offset, sample_interval_ms)
        taper_len = self.config.taper_samples

        if self.config.top_mute:
            result = self._apply_top_mute(result, mute_sample, taper_len, n_samples)

        if self.config.bottom_mute:
            result = self._apply_bottom_mute(result, mute_sample, taper_len, n_samples)

        return result

    def _apply_top_mute(
        self,
        trace: np.ndarray,
        mute_sample: int,
        taper_len: int,
        n_samples: int
    ) -> np.ndarray:
        """
        Apply top mute (zero before mute time, taper after).

        Args:
            trace: Trace array (modified in place)
            mute_sample: Sample index for mute
            taper_len: Number of taper samples
            n_samples: Total samples in trace

        Returns:
            Modified trace
        """
        if mute_sample <= 0:
            return trace

        # Clamp to valid range
        mute_end = min(mute_sample, n_samples)

        # Zero samples before mute
        trace[:mute_end] = 0

        # Apply taper after mute
        if taper_len > 0 and mute_end < n_samples:
            taper_end = min(mute_end + taper_len, n_samples)
            actual_taper_len = taper_end - mute_end

            if actual_taper_len > 0:
                # Use appropriate portion of taper
                taper = self._taper[:actual_taper_len]
                trace[mute_end:taper_end] *= taper

        return trace

    def _apply_bottom_mute(
        self,
        trace: np.ndarray,
        mute_sample: int,
        taper_len: int,
        n_samples: int
    ) -> np.ndarray:
        """
        Apply bottom mute (taper before mute time, zero after).

        Args:
            trace: Trace array (modified in place)
            mute_sample: Sample index for mute
            taper_len: Number of taper samples
            n_samples: Total samples in trace

        Returns:
            Modified trace
        """
        if mute_sample >= n_samples:
            return trace

        # Clamp to valid range
        mute_start = max(0, mute_sample)

        # Apply taper before mute
        if taper_len > 0 and mute_start > 0:
            taper_start = max(0, mute_start - taper_len)
            actual_taper_len = mute_start - taper_start

            if actual_taper_len > 0:
                # Use reversed taper (1 -> 0)
                taper = self._taper[:actual_taper_len][::-1]
                trace[taper_start:mute_start] *= taper

        # Zero samples after mute
        trace[mute_start:] = 0

        return trace

    def apply_mute_batch(
        self,
        traces: np.ndarray,
        offsets: np.ndarray,
        sample_interval_ms: float
    ) -> np.ndarray:
        """
        Apply mute to a batch of traces (vectorized where possible).

        Args:
            traces: 2D array (n_samples, n_traces)
            offsets: 1D array of offsets in meters (n_traces,)
            sample_interval_ms: Sample interval in milliseconds

        Returns:
            Muted traces (copy, input unchanged)
        """
        n_samples, n_traces = traces.shape
        result = traces.copy()

        # Calculate all mute samples at once
        velocity_m_per_ms = self.config.velocity / 1000.0
        mute_times_ms = np.abs(offsets) / velocity_m_per_ms
        mute_samples = (mute_times_ms / sample_interval_ms).astype(np.int32)

        taper_len = self.config.taper_samples

        # Process each trace (can't fully vectorize due to varying mute positions)
        for i in range(n_traces):
            mute_sample = mute_samples[i]

            if self.config.top_mute:
                result[:, i] = self._apply_top_mute(
                    result[:, i], mute_sample, taper_len, n_samples
                )

            if self.config.bottom_mute:
                result[:, i] = self._apply_bottom_mute(
                    result[:, i], mute_sample, taper_len, n_samples
                )

        return result

    def get_description(self) -> str:
        """Get human-readable description."""
        mute_types = []
        if self.config.top_mute:
            mute_types.append("top")
        if self.config.bottom_mute:
            mute_types.append("bottom")

        return (
            f"Linear Mute ({' + '.join(mute_types)}): "
            f"V={self.config.velocity:.0f} m/s, "
            f"taper={self.config.taper_samples} samples"
        )
