"""
Dataset Indexer for Pre-Stack Migration

Scans SEG-Y files and builds trace indices for efficient
bin-based data access during migration.

Features:
- Streaming scan (memory efficient for large files)
- Header extraction and computed header support
- Bin assignment indexing
- Index persistence (save/load)
- Validation and statistics
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Iterator
from dataclasses import dataclass, field
from pathlib import Path
import logging
import json
import time

logger = logging.getLogger(__name__)


@dataclass
class TraceIndexEntry:
    """
    Index entry for a single trace.

    Stores essential information for trace retrieval and bin assignment.
    """
    trace_number: int          # 0-based trace index in file
    file_position: int         # Byte offset in file (for fast access)
    offset: float              # Source-receiver offset (meters)
    azimuth: float             # Source-receiver azimuth (degrees, 0-360)
    inline: Optional[int] = None
    xline: Optional[int] = None
    cdp_x: Optional[float] = None
    cdp_y: Optional[float] = None
    source_x: Optional[float] = None
    source_y: Optional[float] = None
    receiver_x: Optional[float] = None
    receiver_y: Optional[float] = None


@dataclass
class DatasetIndex:
    """
    Complete index of a seismic dataset.

    Contains trace-level index entries and aggregate statistics.
    """
    filepath: str
    n_traces: int
    n_samples: int
    sample_rate_ms: float
    entries: List[TraceIndexEntry] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    creation_time: Optional[str] = None

    def __post_init__(self):
        if self.creation_time is None:
            self.creation_time = time.strftime("%Y-%m-%d %H:%M:%S")

    @property
    def offsets(self) -> np.ndarray:
        """Array of all trace offsets."""
        return np.array([e.offset for e in self.entries], dtype=np.float32)

    @property
    def azimuths(self) -> np.ndarray:
        """Array of all trace azimuths."""
        return np.array([e.azimuth for e in self.entries], dtype=np.float32)

    @property
    def trace_numbers(self) -> np.ndarray:
        """Array of trace numbers."""
        return np.array([e.trace_number for e in self.entries], dtype=np.int32)

    def get_traces_in_offset_range(
        self,
        offset_min: float,
        offset_max: float,
    ) -> np.ndarray:
        """Get trace numbers within offset range."""
        offsets = self.offsets
        mask = (offsets >= offset_min) & (offsets < offset_max)
        return self.trace_numbers[mask]

    def get_traces_for_bin(
        self,
        offset_min: float,
        offset_max: float,
        azimuth_min: float,
        azimuth_max: float,
    ) -> np.ndarray:
        """Get trace numbers matching bin criteria."""
        offsets = self.offsets
        azimuths = self.azimuths

        offset_mask = (offsets >= offset_min) & (offsets < offset_max)

        if azimuth_max < azimuth_min:
            # Wrap-around
            azimuth_mask = (azimuths >= azimuth_min) | (azimuths < azimuth_max)
        else:
            azimuth_mask = (azimuths >= azimuth_min) & (azimuths < azimuth_max)

        return self.trace_numbers[offset_mask & azimuth_mask]

    def compute_statistics(self) -> Dict[str, Any]:
        """Compute and store index statistics."""
        offsets = self.offsets
        azimuths = self.azimuths

        self.statistics = {
            'n_traces': self.n_traces,
            'n_samples': self.n_samples,
            'sample_rate_ms': self.sample_rate_ms,
            'offset_min': float(np.min(offsets)) if len(offsets) > 0 else 0.0,
            'offset_max': float(np.max(offsets)) if len(offsets) > 0 else 0.0,
            'offset_mean': float(np.mean(offsets)) if len(offsets) > 0 else 0.0,
            'offset_std': float(np.std(offsets)) if len(offsets) > 0 else 0.0,
            'azimuth_min': float(np.min(azimuths)) if len(azimuths) > 0 else 0.0,
            'azimuth_max': float(np.max(azimuths)) if len(azimuths) > 0 else 0.0,
        }

        # Check for missing coordinates
        has_source = sum(1 for e in self.entries if e.source_x is not None)
        has_receiver = sum(1 for e in self.entries if e.receiver_x is not None)
        has_inline = sum(1 for e in self.entries if e.inline is not None)

        self.statistics['has_source_coords'] = has_source == self.n_traces
        self.statistics['has_receiver_coords'] = has_receiver == self.n_traces
        self.statistics['has_inline_xline'] = has_inline == self.n_traces

        return self.statistics

    def to_dict(self) -> Dict[str, Any]:
        """Serialize index to dictionary (compact format)."""
        # Store entries as arrays for efficiency
        return {
            'filepath': self.filepath,
            'n_traces': self.n_traces,
            'n_samples': self.n_samples,
            'sample_rate_ms': self.sample_rate_ms,
            'trace_numbers': [e.trace_number for e in self.entries],
            'file_positions': [e.file_position for e in self.entries],
            'offsets': [e.offset for e in self.entries],
            'azimuths': [e.azimuth for e in self.entries],
            'inlines': [e.inline for e in self.entries],
            'xlines': [e.xline for e in self.entries],
            'statistics': self.statistics,
            'metadata': self.metadata,
            'creation_time': self.creation_time,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DatasetIndex':
        """Deserialize from dictionary."""
        n = len(d.get('trace_numbers', []))
        entries = []

        for i in range(n):
            entries.append(TraceIndexEntry(
                trace_number=d['trace_numbers'][i],
                file_position=d['file_positions'][i],
                offset=d['offsets'][i],
                azimuth=d['azimuths'][i],
                inline=d['inlines'][i] if d.get('inlines') else None,
                xline=d['xlines'][i] if d.get('xlines') else None,
            ))

        index = cls(
            filepath=d['filepath'],
            n_traces=d['n_traces'],
            n_samples=d['n_samples'],
            sample_rate_ms=d['sample_rate_ms'],
            entries=entries,
            statistics=d.get('statistics', {}),
            metadata=d.get('metadata', {}),
            creation_time=d.get('creation_time'),
        )
        return index

    def save(self, filepath: str) -> None:
        """Save index to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f)
        logger.info(f"Saved index to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'DatasetIndex':
        """Load index from JSON file."""
        with open(filepath, 'r') as f:
            return cls.from_dict(json.load(f))


class DatasetIndexer:
    """
    Builds trace indices for seismic datasets.

    Scans SEG-Y files and extracts header information for
    efficient bin-based processing.
    """

    def __init__(
        self,
        header_mapping: Optional[Dict[str, str]] = None,
        compute_offset: bool = True,
        compute_azimuth: bool = True,
        progress_callback: Optional[callable] = None,
    ):
        """
        Initialize dataset indexer.

        Args:
            header_mapping: Mapping from required headers to file headers
                           e.g., {'SX': 'SourceX', 'GX': 'GroupX'}
            compute_offset: Compute offset from coordinates if not present
            compute_azimuth: Compute azimuth from coordinates if not present
            progress_callback: Function(current, total, message) for progress
        """
        self.header_mapping = header_mapping or {}
        self.compute_offset = compute_offset
        self.compute_azimuth = compute_azimuth
        self.progress_callback = progress_callback

        # Default SEG-Y header names
        self._default_headers = {
            'source_x': ['SourceX', 'SX', 'sx', 'source_x'],
            'source_y': ['SourceY', 'SY', 'sy', 'source_y'],
            'receiver_x': ['GroupX', 'GX', 'gx', 'receiver_x', 'ReceiverX'],
            'receiver_y': ['GroupY', 'GY', 'gy', 'receiver_y', 'ReceiverY'],
            'offset': ['offset', 'OFFSET', 'Offset'],
            'azimuth': ['azimuth', 'AZIMUTH', 'Azimuth'],
            'inline': ['INLINE_3D', 'inline', 'Inline', 'iline', 'IL'],
            'xline': ['CROSSLINE_3D', 'xline', 'Xline', 'crossline', 'XL'],
        }

    def _report_progress(self, current: int, total: int, message: str = ""):
        """Report progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(current, total, message)

    def _get_header_value(
        self,
        headers: Dict[str, Any],
        field_name: str,
    ) -> Optional[float]:
        """
        Get header value using mapping or default names.

        Args:
            headers: Trace headers dictionary
            field_name: Internal field name (e.g., 'source_x')

        Returns:
            Header value or None if not found
        """
        # Check explicit mapping first
        if field_name in self.header_mapping:
            mapped_name = self.header_mapping[field_name]
            if mapped_name in headers:
                return float(headers[mapped_name])

        # Try default names
        for name in self._default_headers.get(field_name, []):
            if name in headers:
                return float(headers[name])

        return None

    def _compute_offset_azimuth(
        self,
        sx: Optional[float],
        sy: Optional[float],
        gx: Optional[float],
        gy: Optional[float],
    ) -> Tuple[Optional[float], Optional[float]]:
        """Compute offset and azimuth from coordinates."""
        if any(v is None for v in [sx, sy, gx, gy]):
            return None, None

        dx = gx - sx
        dy = gy - sy

        offset = np.sqrt(dx**2 + dy**2)

        # Azimuth: degrees from north (Y-axis), clockwise
        azimuth = np.degrees(np.arctan2(dx, dy))
        if azimuth < 0:
            azimuth += 360

        return float(offset), float(azimuth)

    def index_file(
        self,
        filepath: str,
        max_traces: Optional[int] = None,
    ) -> DatasetIndex:
        """
        Build index for a SEG-Y file.

        Args:
            filepath: Path to SEG-Y file
            max_traces: Maximum traces to index (None = all)

        Returns:
            DatasetIndex with trace information
        """
        import segyio

        logger.info(f"Indexing file: {filepath}")
        start_time = time.time()

        with segyio.open(filepath, 'r', ignore_geometry=True) as f:
            n_traces = f.tracecount
            n_samples = len(f.samples)
            sample_rate = f.bin[segyio.BinField.Interval] / 1000.0  # to ms

            if max_traces:
                n_traces = min(n_traces, max_traces)

            entries = []

            for i in range(n_traces):
                if i % 1000 == 0:
                    self._report_progress(i, n_traces, f"Indexing trace {i}/{n_traces}")

                # Get trace headers as dictionary
                header = f.header[i]
                headers = {str(k): v for k, v in header.items()}

                # Extract coordinates
                sx = self._get_header_value(headers, 'source_x')
                sy = self._get_header_value(headers, 'source_y')
                gx = self._get_header_value(headers, 'receiver_x')
                gy = self._get_header_value(headers, 'receiver_y')

                # Get offset and azimuth (from headers or computed)
                offset = self._get_header_value(headers, 'offset')
                azimuth = self._get_header_value(headers, 'azimuth')

                if self.compute_offset and offset is None:
                    offset, _ = self._compute_offset_azimuth(sx, sy, gx, gy)

                if self.compute_azimuth and azimuth is None:
                    _, azimuth = self._compute_offset_azimuth(sx, sy, gx, gy)

                # Default values if still None
                offset = offset if offset is not None else 0.0
                azimuth = azimuth if azimuth is not None else 0.0

                # Get inline/xline
                inline = self._get_header_value(headers, 'inline')
                xline = self._get_header_value(headers, 'xline')

                # Compute CDP if coordinates available
                cdp_x = (sx + gx) / 2 if sx is not None and gx is not None else None
                cdp_y = (sy + gy) / 2 if sy is not None and gy is not None else None

                entries.append(TraceIndexEntry(
                    trace_number=i,
                    file_position=0,  # Would need file position calculation
                    offset=offset,
                    azimuth=azimuth,
                    inline=int(inline) if inline is not None else None,
                    xline=int(xline) if xline is not None else None,
                    cdp_x=cdp_x,
                    cdp_y=cdp_y,
                    source_x=sx,
                    source_y=sy,
                    receiver_x=gx,
                    receiver_y=gy,
                ))

            self._report_progress(n_traces, n_traces, "Indexing complete")

        elapsed = time.time() - start_time

        index = DatasetIndex(
            filepath=str(filepath),
            n_traces=len(entries),
            n_samples=n_samples,
            sample_rate_ms=sample_rate,
            entries=entries,
            metadata={
                'indexing_time_seconds': elapsed,
                'compute_offset': self.compute_offset,
                'compute_azimuth': self.compute_azimuth,
            }
        )

        index.compute_statistics()

        logger.info(
            f"Indexed {len(entries)} traces in {elapsed:.1f}s "
            f"({len(entries)/elapsed:.0f} traces/s)"
        )

        return index

    def index_from_geometry(
        self,
        n_traces: int,
        n_samples: int,
        sample_rate_ms: float,
        source_x: np.ndarray,
        source_y: np.ndarray,
        receiver_x: np.ndarray,
        receiver_y: np.ndarray,
        inlines: Optional[np.ndarray] = None,
        xlines: Optional[np.ndarray] = None,
    ) -> DatasetIndex:
        """
        Build index from explicit geometry arrays.

        Useful for synthetic data or when geometry is already loaded.

        Args:
            n_traces: Number of traces
            n_samples: Samples per trace
            sample_rate_ms: Sample rate in milliseconds
            source_x, source_y: Source coordinates
            receiver_x, receiver_y: Receiver coordinates
            inlines, xlines: Optional inline/xline numbers

        Returns:
            DatasetIndex
        """
        entries = []

        for i in range(n_traces):
            sx, sy = source_x[i], source_y[i]
            gx, gy = receiver_x[i], receiver_y[i]

            offset, azimuth = self._compute_offset_azimuth(sx, sy, gx, gy)

            entries.append(TraceIndexEntry(
                trace_number=i,
                file_position=0,
                offset=offset if offset is not None else 0.0,
                azimuth=azimuth if azimuth is not None else 0.0,
                inline=int(inlines[i]) if inlines is not None else None,
                xline=int(xlines[i]) if xlines is not None else None,
                cdp_x=(sx + gx) / 2,
                cdp_y=(sy + gy) / 2,
                source_x=float(sx),
                source_y=float(sy),
                receiver_x=float(gx),
                receiver_y=float(gy),
            ))

        index = DatasetIndex(
            filepath="<memory>",
            n_traces=n_traces,
            n_samples=n_samples,
            sample_rate_ms=sample_rate_ms,
            entries=entries,
        )

        index.compute_statistics()
        return index


class BinnedDataset:
    """
    Dataset with pre-computed bin assignments.

    Combines a DatasetIndex with a BinningTable to provide
    efficient bin-based trace access.
    """

    def __init__(
        self,
        index: DatasetIndex,
        binning_table: 'BinningTable',
    ):
        """
        Initialize binned dataset.

        Args:
            index: Dataset index
            binning_table: Binning table defining bins
        """
        from models.binning import BinningTable

        self.index = index
        self.binning_table = binning_table

        # Compute bin assignments
        self._bin_assignments: Dict[str, np.ndarray] = {}
        self._compute_assignments()

    def _compute_assignments(self):
        """Compute trace-to-bin assignments."""
        offsets = self.index.offsets
        azimuths = self.index.azimuths

        self._bin_assignments = self.binning_table.assign_traces_batch(
            offsets, azimuths, enabled_only=True
        )

        logger.info(
            f"Computed bin assignments for {self.index.n_traces} traces "
            f"across {len(self._bin_assignments)} bins"
        )

    def get_bin_trace_numbers(self, bin_name: str) -> np.ndarray:
        """Get trace numbers for a specific bin."""
        if bin_name not in self._bin_assignments:
            return np.array([], dtype=np.int32)

        mask = self._bin_assignments[bin_name]
        return self.index.trace_numbers[mask]

    def get_bin_trace_count(self, bin_name: str) -> int:
        """Get number of traces in a bin."""
        if bin_name not in self._bin_assignments:
            return 0
        return int(np.sum(self._bin_assignments[bin_name]))

    def get_bin_summary(self) -> Dict[str, int]:
        """Get trace counts per bin."""
        return {
            name: self.get_bin_trace_count(name)
            for name in self._bin_assignments
        }

    def iterate_bins(self) -> Iterator[Tuple[str, np.ndarray]]:
        """Iterate over bins, yielding (bin_name, trace_numbers)."""
        for bin_def in self.binning_table.enabled_bins:
            trace_nums = self.get_bin_trace_numbers(bin_def.name)
            if len(trace_nums) > 0:
                yield bin_def.name, trace_nums

    def get_coverage_report(self) -> Dict[str, Any]:
        """Get detailed coverage report."""
        summary = self.get_bin_summary()
        total_assigned = sum(summary.values())

        # Check for overlaps (traces in multiple bins)
        all_masks = list(self._bin_assignments.values())
        if all_masks:
            combined = np.zeros(self.index.n_traces, dtype=int)
            for mask in all_masks:
                combined += mask.astype(int)
            n_multi = int(np.sum(combined > 1))
            n_unassigned = int(np.sum(combined == 0))
        else:
            n_multi = 0
            n_unassigned = self.index.n_traces

        return {
            'n_traces': self.index.n_traces,
            'n_bins': len(summary),
            'bin_counts': summary,
            'total_assigned': total_assigned,
            'n_unassigned': n_unassigned,
            'n_multi_bin': n_multi,
            'coverage_percent': 100.0 * (self.index.n_traces - n_unassigned) / self.index.n_traces
            if self.index.n_traces > 0 else 0.0,
        }


def validate_index(index: DatasetIndex) -> List[str]:
    """
    Validate dataset index for common issues.

    Args:
        index: Dataset index to validate

    Returns:
        List of warning messages (empty if valid)
    """
    warnings = []

    # Check for missing coordinates
    n_missing_source = sum(1 for e in index.entries if e.source_x is None)
    n_missing_receiver = sum(1 for e in index.entries if e.receiver_x is None)

    if n_missing_source > 0:
        warnings.append(f"{n_missing_source} traces missing source coordinates")
    if n_missing_receiver > 0:
        warnings.append(f"{n_missing_receiver} traces missing receiver coordinates")

    # Check for zero/missing offset
    n_zero_offset = sum(1 for e in index.entries if e.offset == 0.0)
    if n_zero_offset > index.n_traces * 0.1:  # More than 10%
        warnings.append(f"{n_zero_offset} traces have zero offset")

    # Check offset range
    offsets = index.offsets
    if len(offsets) > 0:
        if np.max(offsets) > 50000:
            warnings.append(f"Large offset values detected (max={np.max(offsets):.0f}m)")

    # Check azimuth range
    azimuths = index.azimuths
    if len(azimuths) > 0:
        if np.any((azimuths < 0) | (azimuths > 360)):
            warnings.append("Azimuth values outside [0, 360] range")

    return warnings
