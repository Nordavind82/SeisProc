"""
Gather Readers for Seismic Data

Provides streaming access to seismic gathers without loading
entire dataset to memory.

Supported gather types:
- Common Shot
- Common Offset
- Common Receiver
- OVT (Offset Vector Tile)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, List, Tuple, Optional, Dict, Any
import numpy as np
from pathlib import Path

from utils.dataset_indexer import DatasetIndex, BinnedDataset
from utils.sort_detector import SortOrder, get_gather_boundaries


@dataclass
class Gather:
    """A seismic gather (collection of related traces)."""
    gather_id: str
    trace_numbers: np.ndarray  # Original trace indices in dataset
    data: np.ndarray  # Shape: (n_traces, n_samples)
    offsets: np.ndarray  # Shape: (n_traces,)
    azimuths: np.ndarray  # Shape: (n_traces,)
    source_x: Optional[np.ndarray] = None
    source_y: Optional[np.ndarray] = None
    receiver_x: Optional[np.ndarray] = None
    receiver_y: Optional[np.ndarray] = None
    cdp_x: Optional[np.ndarray] = None
    cdp_y: Optional[np.ndarray] = None
    inline: Optional[int] = None
    xline: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

    @property
    def n_traces(self) -> int:
        """Number of traces in gather."""
        return self.data.shape[0]

    @property
    def n_samples(self) -> int:
        """Number of samples per trace."""
        return self.data.shape[1]


class GatherIterator(ABC):
    """
    Abstract base class for gather iterators.

    Provides streaming access to gathers without loading
    entire dataset to memory.
    """

    def __init__(
        self,
        index: DatasetIndex,
        data_reader: 'DataReader',
    ):
        """
        Initialize gather iterator.

        Args:
            index: Dataset index with trace metadata
            data_reader: Reader for accessing trace data
        """
        self.index = index
        self.data_reader = data_reader
        self._current_position = 0

    @abstractmethod
    def __iter__(self) -> Iterator[Gather]:
        """Iterate over gathers."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return total number of gathers."""
        pass

    @abstractmethod
    def get_gather(self, gather_id: str) -> Gather:
        """Get a specific gather by ID."""
        pass

    @abstractmethod
    def get_gather_ids(self) -> List[str]:
        """Get list of all gather IDs."""
        pass

    def reset(self):
        """Reset iterator to beginning."""
        self._current_position = 0


class DataReader(ABC):
    """Abstract interface for reading trace data."""

    @abstractmethod
    def read_traces(self, trace_numbers: np.ndarray) -> np.ndarray:
        """
        Read specified traces from data source.

        Args:
            trace_numbers: Array of trace indices to read

        Returns:
            Array of shape (n_traces, n_samples)
        """
        pass

    @abstractmethod
    def get_n_samples(self) -> int:
        """Get number of samples per trace."""
        pass


class NumpyDataReader(DataReader):
    """Data reader for numpy arrays (for testing)."""

    def __init__(self, data: np.ndarray):
        """
        Initialize with numpy array.

        Args:
            data: Array of shape (n_traces, n_samples)
        """
        self._data = data

    def read_traces(self, trace_numbers: np.ndarray) -> np.ndarray:
        """Read traces from numpy array."""
        return self._data[trace_numbers]

    def get_n_samples(self) -> int:
        """Get number of samples per trace."""
        return self._data.shape[1]


class CommonOffsetGatherIterator(GatherIterator):
    """
    Iterator for common offset gathers.

    Groups traces by offset bin and iterates through each bin.
    """

    def __init__(
        self,
        index: DatasetIndex,
        data_reader: DataReader,
        binned_dataset: BinnedDataset,
    ):
        """
        Initialize common offset gather iterator.

        Args:
            index: Dataset index
            data_reader: Data reader
            binned_dataset: Binned dataset with offset assignments
        """
        super().__init__(index, data_reader)
        self.binned_dataset = binned_dataset
        self._gather_ids = list(binned_dataset._bin_assignments.keys())

    def __iter__(self) -> Iterator[Gather]:
        """Iterate over offset gathers."""
        for bin_name, trace_numbers in self.binned_dataset.iterate_bins():
            yield self._create_gather(bin_name, trace_numbers)

    def __len__(self) -> int:
        """Return number of offset bins."""
        return len(self._gather_ids)

    def get_gather(self, gather_id: str) -> Gather:
        """Get gather for specific offset bin."""
        trace_numbers = self.binned_dataset.get_bin_trace_numbers(gather_id)
        return self._create_gather(gather_id, trace_numbers)

    def get_gather_ids(self) -> List[str]:
        """Get list of offset bin names."""
        return self._gather_ids.copy()

    def _create_gather(self, bin_name: str, trace_numbers: List[int]) -> Gather:
        """Create gather from bin name and trace numbers."""
        trace_arr = np.array(trace_numbers, dtype=np.int32)

        # Read trace data
        data = self.data_reader.read_traces(trace_arr)

        # Extract geometry from index
        entries = [self.index.entries[t] for t in trace_numbers]

        offsets = np.array([e.offset for e in entries], dtype=np.float32)
        azimuths = np.array([e.azimuth for e in entries], dtype=np.float32)

        # Optional geometry
        source_x = None
        source_y = None
        receiver_x = None
        receiver_y = None
        cdp_x = None
        cdp_y = None

        if entries[0].source_x is not None:
            source_x = np.array([e.source_x for e in entries], dtype=np.float32)
            source_y = np.array([e.source_y for e in entries], dtype=np.float32)
            receiver_x = np.array([e.receiver_x for e in entries], dtype=np.float32)
            receiver_y = np.array([e.receiver_y for e in entries], dtype=np.float32)

        if entries[0].cdp_x is not None:
            cdp_x = np.array([e.cdp_x for e in entries], dtype=np.float32)
            cdp_y = np.array([e.cdp_y for e in entries], dtype=np.float32)

        # Get bin info for metadata
        bin_obj = self.binned_dataset.binning_table.get_bin(bin_name)

        return Gather(
            gather_id=bin_name,
            trace_numbers=trace_arr,
            data=data,
            offsets=offsets,
            azimuths=azimuths,
            source_x=source_x,
            source_y=source_y,
            receiver_x=receiver_x,
            receiver_y=receiver_y,
            cdp_x=cdp_x,
            cdp_y=cdp_y,
            metadata={
                'bin_offset_min': bin_obj.offset_min,
                'bin_offset_max': bin_obj.offset_max,
                'bin_offset_center': (bin_obj.offset_min + bin_obj.offset_max) / 2,
                'bin_azimuth_min': bin_obj.azimuth_min,
                'bin_azimuth_max': bin_obj.azimuth_max,
            }
        )


class CommonShotGatherIterator(GatherIterator):
    """
    Iterator for common shot gathers.

    Groups traces by shot ID.
    """

    def __init__(
        self,
        index: DatasetIndex,
        data_reader: DataReader,
        shot_ids: Optional[np.ndarray] = None,
    ):
        """
        Initialize common shot gather iterator.

        Args:
            index: Dataset index
            data_reader: Data reader
            shot_ids: Shot IDs per trace (if not in index)
        """
        super().__init__(index, data_reader)

        # Get shot IDs from index or provided array
        if shot_ids is not None:
            self._shot_ids = shot_ids
        else:
            # Try to extract from index entries
            self._shot_ids = self._extract_shot_ids()

        # Build shot gather mapping
        self._shot_traces = self._build_shot_mapping()
        self._gather_ids = list(self._shot_traces.keys())

    def _extract_shot_ids(self) -> np.ndarray:
        """Extract shot IDs from index entries."""
        # Use source coordinates as proxy for shot ID if available
        shot_ids = []
        for entry in self.index.entries:
            if entry.source_x is not None and entry.source_y is not None:
                # Create shot ID from source position
                shot_id = int(entry.source_x * 1000 + entry.source_y)
            else:
                # Default to trace number
                shot_id = entry.trace_number
            shot_ids.append(shot_id)
        return np.array(shot_ids, dtype=np.int64)

    def _build_shot_mapping(self) -> Dict[str, List[int]]:
        """Build mapping from shot ID to trace numbers."""
        shot_traces = {}
        for i, shot_id in enumerate(self._shot_ids):
            key = f"shot_{shot_id}"
            if key not in shot_traces:
                shot_traces[key] = []
            shot_traces[key].append(i)
        return shot_traces

    def __iter__(self) -> Iterator[Gather]:
        """Iterate over shot gathers."""
        for shot_id, trace_numbers in self._shot_traces.items():
            yield self._create_gather(shot_id, trace_numbers)

    def __len__(self) -> int:
        """Return number of shots."""
        return len(self._gather_ids)

    def get_gather(self, gather_id: str) -> Gather:
        """Get gather for specific shot."""
        trace_numbers = self._shot_traces[gather_id]
        return self._create_gather(gather_id, trace_numbers)

    def get_gather_ids(self) -> List[str]:
        """Get list of shot IDs."""
        return self._gather_ids.copy()

    def _create_gather(self, shot_id: str, trace_numbers: List[int]) -> Gather:
        """Create gather from shot ID and trace numbers."""
        trace_arr = np.array(trace_numbers, dtype=np.int32)
        data = self.data_reader.read_traces(trace_arr)

        entries = [self.index.entries[t] for t in trace_numbers]
        offsets = np.array([e.offset for e in entries], dtype=np.float32)
        azimuths = np.array([e.azimuth for e in entries], dtype=np.float32)

        source_x = None
        source_y = None
        receiver_x = None
        receiver_y = None

        if entries[0].source_x is not None:
            source_x = np.array([e.source_x for e in entries], dtype=np.float32)
            source_y = np.array([e.source_y for e in entries], dtype=np.float32)
            receiver_x = np.array([e.receiver_x for e in entries], dtype=np.float32)
            receiver_y = np.array([e.receiver_y for e in entries], dtype=np.float32)

        return Gather(
            gather_id=shot_id,
            trace_numbers=trace_arr,
            data=data,
            offsets=offsets,
            azimuths=azimuths,
            source_x=source_x,
            source_y=source_y,
            receiver_x=receiver_x,
            receiver_y=receiver_y,
        )


class OVTGatherIterator(GatherIterator):
    """
    Iterator for Offset Vector Tile (OVT) gathers.

    Groups traces by offset and azimuth sectors.
    """

    def __init__(
        self,
        index: DatasetIndex,
        data_reader: DataReader,
        binned_dataset: BinnedDataset,
    ):
        """
        Initialize OVT gather iterator.

        Args:
            index: Dataset index
            data_reader: Data reader
            binned_dataset: Binned dataset with OVT assignments
        """
        super().__init__(index, data_reader)
        self.binned_dataset = binned_dataset
        self._gather_ids = list(binned_dataset._bin_assignments.keys())

    def __iter__(self) -> Iterator[Gather]:
        """Iterate over OVT gathers."""
        for bin_name, trace_numbers in self.binned_dataset.iterate_bins():
            yield self._create_gather(bin_name, trace_numbers)

    def __len__(self) -> int:
        """Return number of OVT bins."""
        return len(self._gather_ids)

    def get_gather(self, gather_id: str) -> Gather:
        """Get gather for specific OVT bin."""
        trace_numbers = self.binned_dataset.get_bin_trace_numbers(gather_id)
        return self._create_gather(gather_id, trace_numbers)

    def get_gather_ids(self) -> List[str]:
        """Get list of OVT bin names."""
        return self._gather_ids.copy()

    def _create_gather(self, bin_name: str, trace_numbers: List[int]) -> Gather:
        """Create gather from OVT bin."""
        trace_arr = np.array(trace_numbers, dtype=np.int32)
        data = self.data_reader.read_traces(trace_arr)

        entries = [self.index.entries[t] for t in trace_numbers]
        offsets = np.array([e.offset for e in entries], dtype=np.float32)
        azimuths = np.array([e.azimuth for e in entries], dtype=np.float32)

        source_x = None
        source_y = None
        receiver_x = None
        receiver_y = None
        cdp_x = None
        cdp_y = None

        if entries[0].source_x is not None:
            source_x = np.array([e.source_x for e in entries], dtype=np.float32)
            source_y = np.array([e.source_y for e in entries], dtype=np.float32)
            receiver_x = np.array([e.receiver_x for e in entries], dtype=np.float32)
            receiver_y = np.array([e.receiver_y for e in entries], dtype=np.float32)

        if entries[0].cdp_x is not None:
            cdp_x = np.array([e.cdp_x for e in entries], dtype=np.float32)
            cdp_y = np.array([e.cdp_y for e in entries], dtype=np.float32)

        bin_obj = self.binned_dataset.binning_table.get_bin(bin_name)

        return Gather(
            gather_id=bin_name,
            trace_numbers=trace_arr,
            data=data,
            offsets=offsets,
            azimuths=azimuths,
            source_x=source_x,
            source_y=source_y,
            receiver_x=receiver_x,
            receiver_y=receiver_y,
            cdp_x=cdp_x,
            cdp_y=cdp_y,
            metadata={
                'bin_offset_min': bin_obj.offset_min,
                'bin_offset_max': bin_obj.offset_max,
                'bin_offset_center': (bin_obj.offset_min + bin_obj.offset_max) / 2,
                'bin_azimuth_min': bin_obj.azimuth_min,
                'bin_azimuth_max': bin_obj.azimuth_max,
                'bin_azimuth_center': (bin_obj.azimuth_min + bin_obj.azimuth_max) / 2,
            }
        )


class StreamingGatherReader:
    """
    High-level interface for streaming gather access.

    Automatically selects appropriate iterator based on
    gather type and data organization.
    """

    def __init__(
        self,
        index: DatasetIndex,
        data_reader: DataReader,
        binned_dataset: Optional[BinnedDataset] = None,
        shot_ids: Optional[np.ndarray] = None,
    ):
        """
        Initialize streaming gather reader.

        Args:
            index: Dataset index
            data_reader: Data reader
            binned_dataset: Binned dataset (for offset/OVT gathers)
            shot_ids: Shot IDs (for shot gathers)
        """
        self.index = index
        self.data_reader = data_reader
        self.binned_dataset = binned_dataset
        self.shot_ids = shot_ids

    def iter_offset_gathers(self) -> Iterator[Gather]:
        """
        Iterate over common offset gathers.

        Requires binned_dataset to be set.
        """
        if self.binned_dataset is None:
            raise ValueError("binned_dataset required for offset gather iteration")

        iterator = CommonOffsetGatherIterator(
            self.index, self.data_reader, self.binned_dataset
        )
        return iter(iterator)

    def iter_shot_gathers(self) -> Iterator[Gather]:
        """Iterate over common shot gathers."""
        iterator = CommonShotGatherIterator(
            self.index, self.data_reader, self.shot_ids
        )
        return iter(iterator)

    def iter_ovt_gathers(self) -> Iterator[Gather]:
        """
        Iterate over OVT gathers.

        Requires binned_dataset with OVT binning to be set.
        """
        if self.binned_dataset is None:
            raise ValueError("binned_dataset required for OVT gather iteration")

        iterator = OVTGatherIterator(
            self.index, self.data_reader, self.binned_dataset
        )
        return iter(iterator)

    def get_gather_count(self, gather_type: str) -> int:
        """
        Get number of gathers for given type.

        Args:
            gather_type: "offset", "shot", or "ovt"

        Returns:
            Number of gathers
        """
        if gather_type == "offset":
            if self.binned_dataset is None:
                return 0
            return len(self.binned_dataset._bin_assignments)

        elif gather_type == "shot":
            iterator = CommonShotGatherIterator(
                self.index, self.data_reader, self.shot_ids
            )
            return len(iterator)

        elif gather_type == "ovt":
            if self.binned_dataset is None:
                return 0
            return len(self.binned_dataset._bin_assignments)

        else:
            raise ValueError(f"Unknown gather type: {gather_type}")


def create_gather_iterator(
    index: DatasetIndex,
    data_reader: DataReader,
    gather_type: str,
    binned_dataset: Optional[BinnedDataset] = None,
    shot_ids: Optional[np.ndarray] = None,
) -> GatherIterator:
    """
    Factory function to create appropriate gather iterator.

    Args:
        index: Dataset index
        data_reader: Data reader
        gather_type: Type of gather ("offset", "shot", "ovt")
        binned_dataset: Binned dataset (required for offset/ovt)
        shot_ids: Shot IDs (optional for shot gathers)

    Returns:
        Appropriate GatherIterator subclass
    """
    if gather_type == "offset":
        if binned_dataset is None:
            raise ValueError("binned_dataset required for offset gathers")
        return CommonOffsetGatherIterator(index, data_reader, binned_dataset)

    elif gather_type == "shot":
        return CommonShotGatherIterator(index, data_reader, shot_ids)

    elif gather_type == "ovt":
        if binned_dataset is None:
            raise ValueError("binned_dataset required for OVT gathers")
        return OVTGatherIterator(index, data_reader, binned_dataset)

    else:
        raise ValueError(f"Unknown gather type: {gather_type}")
