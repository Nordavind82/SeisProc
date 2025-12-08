"""
Smart partitioner for dividing SEGY files into segments at gather boundaries.

Uses quick boundary probing instead of full header scan.
"""

import segyio
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


@dataclass
class Segment:
    """Represents a segment of traces to be processed by one worker."""
    segment_id: int
    start_trace: int      # Inclusive
    end_trace: int        # Exclusive
    n_traces: int

    def __post_init__(self):
        if self.n_traces != self.end_trace - self.start_trace:
            self.n_traces = self.end_trace - self.start_trace


@dataclass
class PartitionConfig:
    """Configuration for partitioning."""
    segy_path: str
    n_segments: int
    total_traces: int
    ensemble_key: str = 'cdp'           # Header field for gather detection
    ensemble_byte: int = 21             # Byte location for ensemble key
    max_probe_distance: int = 10000     # Max traces to search for boundary


class SmartPartitioner:
    """
    Partitions SEGY file into segments at gather boundaries.

    Uses quick boundary probing - only reads ~100 headers per split point
    instead of scanning all headers. For 14 workers, reads ~1400 headers
    total (< 1 second) instead of millions.
    """

    def __init__(self, config: PartitionConfig):
        """
        Initialize partitioner.

        Args:
            config: Partition configuration
        """
        self.config = config
        self.segy_path = Path(config.segy_path)

        if not self.segy_path.exists():
            raise FileNotFoundError(f"SEGY file not found: {self.segy_path}")

    def partition(self) -> List[Segment]:
        """
        Partition file into segments at gather boundaries.

        Returns:
            List of Segment objects with adjusted boundaries
        """
        n_traces = self.config.total_traces
        n_segments = self.config.n_segments

        # Handle edge case: fewer traces than segments
        if n_traces < n_segments:
            return [Segment(
                segment_id=0,
                start_trace=0,
                end_trace=n_traces,
                n_traces=n_traces
            )]

        # Calculate raw split points
        traces_per_segment = n_traces // n_segments
        raw_splits = [i * traces_per_segment for i in range(n_segments)]
        raw_splits.append(n_traces)  # End of last segment

        # Adjust split points to gather boundaries
        if self.config.ensemble_key:
            adjusted_splits = self._adjust_splits_to_boundaries(raw_splits)
        else:
            # No ensemble key - use raw splits
            adjusted_splits = raw_splits

        # Create segments
        segments = []
        for i in range(n_segments):
            start = adjusted_splits[i]
            end = adjusted_splits[i + 1]

            # Skip empty segments (can happen with large gathers)
            if end > start:
                segments.append(Segment(
                    segment_id=i,
                    start_trace=start,
                    end_trace=end,
                    n_traces=end - start
                ))

        return segments

    def _adjust_splits_to_boundaries(self, raw_splits: List[int]) -> List[int]:
        """
        Adjust split points to land on gather boundaries.

        Args:
            raw_splits: List of raw split point indices

        Returns:
            Adjusted split points at gather boundaries
        """
        adjusted = [0]  # First segment always starts at 0

        with segyio.open(str(self.segy_path), 'r', ignore_geometry=True) as f:
            f.mmap()

            # Adjust each internal split point
            for i in range(1, len(raw_splits) - 1):
                raw_point = raw_splits[i]
                adjusted_point = self._find_gather_boundary(f, raw_point)

                # Ensure we don't go backwards past previous split
                if adjusted_point <= adjusted[-1]:
                    adjusted_point = adjusted[-1] + 1

                adjusted.append(adjusted_point)

        # Last split is always at end
        adjusted.append(raw_splits[-1])

        return adjusted

    def _find_gather_boundary(self, f, approx_trace: int) -> int:
        """
        Find gather boundary near approximate trace index.

        Probes backward from approx_trace to find where the gather starts.
        Returns the first trace of the current gather.

        Args:
            f: Open segyio file handle
            approx_trace: Approximate split point

        Returns:
            Trace index at gather boundary
        """
        byte_loc = self.config.ensemble_byte
        max_probe = self.config.max_probe_distance

        # Clamp to valid range
        approx_trace = min(approx_trace, f.tracecount - 1)
        approx_trace = max(approx_trace, 0)

        # Get value at approximate point
        current_value = f.attributes(byte_loc)[approx_trace][0]

        # Probe backward to find where this gather starts
        boundary = approx_trace
        probe_count = 0

        while boundary > 0 and probe_count < max_probe:
            prev_value = f.attributes(byte_loc)[boundary - 1][0]

            if prev_value != current_value:
                # Found boundary - current trace is start of new gather
                break

            boundary -= 1
            probe_count += 1

        return boundary

    def get_partition_stats(self, segments: List[Segment]) -> dict:
        """
        Get statistics about the partition.

        Args:
            segments: List of segments

        Returns:
            Dictionary with partition statistics
        """
        if not segments:
            return {}

        trace_counts = [s.n_traces for s in segments]
        total = sum(trace_counts)

        return {
            'n_segments': len(segments),
            'total_traces': total,
            'min_segment': min(trace_counts),
            'max_segment': max(trace_counts),
            'avg_segment': total / len(segments),
            'imbalance_pct': (max(trace_counts) - min(trace_counts)) / (total / len(segments)) * 100
        }


def get_ensemble_byte_location(ensemble_key: str) -> int:
    """
    Get byte location for common ensemble keys.

    Args:
        ensemble_key: Header field name

    Returns:
        Byte location in trace header
    """
    key_to_byte = {
        'cdp': 21,
        'CDP': 21,
        'ffid': 9,
        'FFID': 9,
        'fldr': 9,
        'FieldRecord': 9,
        'shot': 17,
        'ep': 17,
        'EnergySourcePoint': 17,
        'inline': 189,
        'INLINE_3D': 189,
        'crossline': 193,
        'CROSSLINE_3D': 193,
        'offset': 37,
    }

    return key_to_byte.get(ensemble_key, 21)  # Default to CDP
