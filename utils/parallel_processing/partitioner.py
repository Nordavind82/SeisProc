"""
Gather partitioner for parallel processing.

Divides gathers into segments for worker processes, balancing by trace count
to ensure even workload distribution.
"""

import pandas as pd
from typing import List
from .config import GatherSegment


class GatherPartitioner:
    """
    Partitions gathers into segments for parallel processing.

    Balances segments by total trace count rather than gather count,
    ensuring workers have roughly equal workloads even when gather
    sizes vary significantly.
    """

    def __init__(self, ensemble_df: pd.DataFrame, n_segments: int):
        """
        Initialize partitioner.

        Args:
            ensemble_df: DataFrame with ensemble info (must have columns:
                        'ensemble_id', 'start_trace', 'end_trace', 'n_traces')
            n_segments: Number of segments to create (typically = n_workers)
        """
        self.ensemble_df = ensemble_df.reset_index(drop=True)
        self.n_segments = n_segments
        self.n_gathers = len(ensemble_df)
        self.total_traces = ensemble_df['n_traces'].sum()

    def partition(self) -> List[GatherSegment]:
        """
        Partition gathers into segments balanced by trace count.

        Returns:
            List of GatherSegment objects
        """
        if self.n_gathers == 0:
            return []

        if self.n_segments >= self.n_gathers:
            # More workers than gathers - one gather per segment
            return self._partition_one_per_segment()

        return self._partition_by_traces()

    def _partition_one_per_segment(self) -> List[GatherSegment]:
        """Create one segment per gather when n_segments >= n_gathers."""
        segments = []

        for i, row in self.ensemble_df.iterrows():
            segments.append(GatherSegment(
                segment_id=i,
                start_gather=i,
                end_gather=i,
                start_trace=int(row['start_trace']),
                end_trace=int(row['end_trace']),
                n_gathers=1,
                n_traces=int(row['n_traces'])
            ))

        return segments

    def _partition_by_traces(self) -> List[GatherSegment]:
        """
        Partition gathers so each segment has roughly equal trace count.

        Algorithm:
        1. Calculate target traces per segment
        2. Accumulate gathers until target reached
        3. Start new segment at gather boundary
        4. Last segment gets remainder
        """
        target_per_segment = self.total_traces / self.n_segments
        segments = []

        current_segment_traces = 0
        segment_start_gather = 0
        segment_start_trace = int(self.ensemble_df.iloc[0]['start_trace'])

        for i, row in self.ensemble_df.iterrows():
            current_segment_traces += int(row['n_traces'])

            # Check if we should end this segment
            # Don't end if this is the last segment (let it accumulate remainder)
            should_end_segment = (
                current_segment_traces >= target_per_segment and
                len(segments) < self.n_segments - 1 and
                i < self.n_gathers - 1  # Not the last gather
            )

            if should_end_segment:
                # End current segment at this gather
                segments.append(GatherSegment(
                    segment_id=len(segments),
                    start_gather=segment_start_gather,
                    end_gather=i,
                    start_trace=segment_start_trace,
                    end_trace=int(row['end_trace']),
                    n_gathers=i - segment_start_gather + 1,
                    n_traces=current_segment_traces
                ))

                # Start new segment
                segment_start_gather = i + 1
                if i + 1 < self.n_gathers:
                    segment_start_trace = int(self.ensemble_df.iloc[i + 1]['start_trace'])
                current_segment_traces = 0

        # Last segment gets remainder
        if segment_start_gather < self.n_gathers:
            last_row = self.ensemble_df.iloc[-1]
            remaining_traces = sum(
                int(self.ensemble_df.iloc[j]['n_traces'])
                for j in range(segment_start_gather, self.n_gathers)
            )

            segments.append(GatherSegment(
                segment_id=len(segments),
                start_gather=segment_start_gather,
                end_gather=self.n_gathers - 1,
                start_trace=segment_start_trace,
                end_trace=int(last_row['end_trace']),
                n_gathers=self.n_gathers - segment_start_gather,
                n_traces=remaining_traces
            ))

        return segments

    def get_partition_stats(self, segments: List[GatherSegment]) -> dict:
        """
        Get statistics about the partition.

        Args:
            segments: List of segments from partition()

        Returns:
            Dictionary with partition statistics
        """
        if not segments:
            return {
                'n_segments': 0,
                'total_gathers': 0,
                'total_traces': 0
            }

        trace_counts = [s.n_traces for s in segments]
        gather_counts = [s.n_gathers for s in segments]

        return {
            'n_segments': len(segments),
            'total_gathers': sum(gather_counts),
            'total_traces': sum(trace_counts),
            'min_traces_per_segment': min(trace_counts),
            'max_traces_per_segment': max(trace_counts),
            'avg_traces_per_segment': sum(trace_counts) / len(segments),
            'trace_imbalance_pct': (max(trace_counts) - min(trace_counts)) / max(trace_counts) * 100,
            'min_gathers_per_segment': min(gather_counts),
            'max_gathers_per_segment': max(gather_counts),
        }
