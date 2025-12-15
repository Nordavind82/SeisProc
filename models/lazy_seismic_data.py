"""
Lazy seismic data model - memory-efficient wrapper for large datasets.

Provides SeismicData-compatible interface while loading data on-demand from Zarr storage.
"""
import numpy as np
import zarr
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import json

from utils.parquet_io import read_parquet


class LazySeismicData:
    """
    Memory-efficient wrapper for large seismic datasets stored in Zarr format.

    This class provides a SeismicData-compatible interface while loading data on-demand
    from Zarr storage, enabling work with datasets much larger than available RAM.

    Key Features:
    - Constant memory footprint (~10 MB) regardless of dataset size
    - On-demand data loading with windowing support
    - Compatible with existing SeismicData interface (where applicable)
    - Read-only access to prevent accidental modifications

    Storage Structure:
        data_dir/
            ├── traces.zarr/           # Memory-mapped trace data
            ├── headers.parquet        # Headers (loaded on demand)
            ├── ensemble_index.parquet # Ensemble boundaries
            └── metadata.json          # Dataset metadata

    Example:
        >>> lazy_data = LazySeismicData.from_storage_dir('/path/to/zarr_data')
        >>> print(lazy_data.n_traces, lazy_data.n_samples)
        >>> window = lazy_data.get_window(0, 1000, 0, 100)  # Load subset
        >>> ensemble = lazy_data.get_ensemble(5)  # Load specific ensemble
    """

    def __init__(self, zarr_path: Path, metadata: Dict[str, Any],
                 headers_path: Optional[Path] = None,
                 ensemble_index_path: Optional[Path] = None):
        """
        Initialize lazy seismic data wrapper.

        Args:
            zarr_path: Path to traces.zarr directory
            metadata: Dataset metadata dictionary
            headers_path: Optional path to headers.parquet
            ensemble_index_path: Optional path to ensemble_index.parquet
        """
        self.zarr_path = Path(zarr_path)
        self.metadata = metadata
        self.headers_path = headers_path
        self.ensemble_index_path = ensemble_index_path

        # Open Zarr array in read-only mode (memory-mapped, not loaded)
        self._zarr_array = zarr.open_array(str(self.zarr_path), mode='r')

        # Cache ensemble index if available (small, can keep in memory)
        # Using Polars for faster loading when available
        self._ensemble_index = None
        if ensemble_index_path and ensemble_index_path.exists():
            self._ensemble_index = read_parquet(ensemble_index_path)

        # Store metadata for quick access
        self._n_samples = int(metadata.get('n_samples', self._zarr_array.shape[0]))
        self._n_traces = int(metadata.get('n_traces', self._zarr_array.shape[1]))
        self._sample_rate = float(metadata.get('sample_rate', 2.0))

    @classmethod
    def from_storage_dir(cls, storage_dir: str) -> 'LazySeismicData':
        """
        Create LazySeismicData from a storage directory.

        Args:
            storage_dir: Path to directory containing Zarr and Parquet files

        Returns:
            LazySeismicData instance

        Raises:
            FileNotFoundError: If required files don't exist
            ValueError: If metadata is invalid
        """
        storage_path = Path(storage_dir)

        # Handle case where user selected traces.zarr or noise.zarr directory directly
        if storage_path.name in ('traces.zarr', 'noise.zarr'):
            storage_path = storage_path.parent

        # Check required files - try traces.zarr first, fall back to noise.zarr
        zarr_path = storage_path / 'traces.zarr'
        if not zarr_path.exists():
            # Fall back to noise.zarr for noise-only output datasets
            zarr_path = storage_path / 'noise.zarr'
        metadata_path = storage_path / 'metadata.json'

        if not zarr_path.exists():
            raise FileNotFoundError(f"Zarr array not found: {storage_path / 'traces.zarr'}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Optional files
        headers_path = storage_path / 'headers.parquet'
        ensemble_index_path = storage_path / 'ensemble_index.parquet'

        headers_path = headers_path if headers_path.exists() else None
        ensemble_index_path = ensemble_index_path if ensemble_index_path.exists() else None

        return cls(zarr_path, metadata, headers_path, ensemble_index_path)

    # Properties matching SeismicData interface
    @property
    def n_samples(self) -> int:
        """Number of time samples per trace."""
        return self._n_samples

    @property
    def n_traces(self) -> int:
        """Number of traces in dataset."""
        return self._n_traces

    @property
    def sample_rate(self) -> float:
        """Sample rate in milliseconds."""
        return self._sample_rate

    @property
    def duration(self) -> float:
        """Total duration in milliseconds."""
        return (self.n_samples - 1) * self.sample_rate

    @property
    def nyquist_freq(self) -> float:
        """Nyquist frequency in Hz."""
        return 1000.0 / (2.0 * self.sample_rate)

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of dataset (n_samples, n_traces)."""
        return (self.n_samples, self.n_traces)

    def get_time_axis(self) -> np.ndarray:
        """Get time axis in milliseconds."""
        return np.arange(self.n_samples) * self.sample_rate

    def get_trace_axis(self) -> np.ndarray:
        """Get trace axis (trace numbers)."""
        return np.arange(self.n_traces)

    # Data access methods (load on-demand)
    def get_window(self, time_start: float, time_end: float,
                   trace_start: int, trace_end: int) -> np.ndarray:
        """
        Load a rectangular window of data.

        This is the primary data access method, loading only the requested window
        from Zarr storage into memory.

        Args:
            time_start: Start time in milliseconds
            time_end: End time in milliseconds
            trace_start: Start trace index (0-based)
            trace_end: End trace index (exclusive)

        Returns:
            numpy array of shape (n_samples_window, n_traces_window)

        Example:
            >>> data = lazy_data.get_window(0, 1000, 0, 100)
            >>> print(data.shape)  # (500, 100) for 2ms sampling
        """
        # Convert time to sample indices
        sample_start = int(time_start / self.sample_rate)
        sample_end = int(time_end / self.sample_rate)

        # Clip to valid ranges
        sample_start = max(0, min(sample_start, self.n_samples - 1))
        sample_end = max(sample_start + 1, min(sample_end, self.n_samples))
        trace_start = max(0, min(trace_start, self.n_traces - 1))
        trace_end = max(trace_start + 1, min(trace_end, self.n_traces))

        # Load window from Zarr (this is the only actual I/O)
        window_data = self._zarr_array[sample_start:sample_end, trace_start:trace_end]

        # Return as numpy array (copy from memory-mapped array)
        return np.array(window_data)

    def get_trace_range(self, trace_start: int, trace_end: int) -> np.ndarray:
        """
        Load a range of traces (all samples).

        Args:
            trace_start: Start trace index (0-based)
            trace_end: End trace index (exclusive)

        Returns:
            numpy array of shape (n_samples, n_traces_in_range)

        Example:
            >>> traces = lazy_data.get_trace_range(100, 200)
            >>> print(traces.shape)  # (n_samples, 100)
        """
        # Use time that includes all samples (not duration which is n_samples-1)
        return self.get_window(0, self.n_samples * self.sample_rate, trace_start, trace_end)

    def get_time_range(self, time_start: float, time_end: float) -> np.ndarray:
        """
        Load a time range (all traces).

        Args:
            time_start: Start time in milliseconds
            time_end: End time in milliseconds

        Returns:
            numpy array of shape (n_samples_in_range, n_traces)

        Example:
            >>> time_slice = lazy_data.get_time_range(500, 1500)
        """
        # Pass trace range that includes all traces
        return self.get_window(time_start, time_end, 0, self.n_traces)

    def get_ensemble(self, ensemble_id: int) -> np.ndarray:
        """
        Load a specific ensemble (gather).

        Args:
            ensemble_id: Ensemble identifier (0-based)

        Returns:
            numpy array of shape (n_samples, n_traces_in_ensemble)

        Raises:
            ValueError: If ensemble_id is invalid or ensemble index not available

        Example:
            >>> ensemble = lazy_data.get_ensemble(10)
            >>> print(ensemble.shape)  # (n_samples, traces_in_ensemble_10)
        """
        if self._ensemble_index is None:
            raise ValueError("Ensemble index not available for this dataset")

        if ensemble_id < 0 or ensemble_id >= len(self._ensemble_index):
            raise ValueError(f"Invalid ensemble_id: {ensemble_id} "
                           f"(valid range: 0-{len(self._ensemble_index)-1})")

        # Get ensemble boundaries from index
        ensemble_row = self._ensemble_index.iloc[ensemble_id]
        start_trace = int(ensemble_row['start_trace'])
        end_trace = int(ensemble_row['end_trace'])

        # Load traces for this ensemble (all samples)
        return self.get_trace_range(start_trace, end_trace + 1)

    def get_ensemble_count(self) -> int:
        """
        Get number of ensembles.

        Returns:
            Number of ensembles, or 0 if ensemble index not available
        """
        if self._ensemble_index is None:
            return 0
        return len(self._ensemble_index)

    def get_ensemble_info(self, ensemble_id: int) -> Dict[str, Any]:
        """
        Get metadata about an ensemble without loading traces.

        Args:
            ensemble_id: Ensemble identifier (0-based)

        Returns:
            Dictionary with ensemble information (id, start_trace, end_trace, n_traces, etc.)

        Raises:
            ValueError: If ensemble_id is invalid or ensemble index not available
        """
        if self._ensemble_index is None:
            raise ValueError("Ensemble index not available for this dataset")

        if ensemble_id < 0 or ensemble_id >= len(self._ensemble_index):
            raise ValueError(f"Invalid ensemble_id: {ensemble_id}")

        ensemble_row = self._ensemble_index.iloc[ensemble_id]
        return ensemble_row.to_dict()

    def get_headers(self, trace_indices: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Load trace headers.

        Args:
            trace_indices: Optional array of specific trace indices to load.
                          If None, returns all headers (may be large!)

        Returns:
            DataFrame with trace headers

        Warning:
            Loading all headers for large datasets may use significant memory.
            Use trace_indices to load specific headers when possible.

        Example:
            >>> # Load headers for specific traces
            >>> headers = lazy_data.get_headers(np.array([0, 100, 200]))
            >>> # Load all headers (use with caution for large datasets)
            >>> all_headers = lazy_data.get_headers()
        """
        if self.headers_path is None:
            raise ValueError("Headers not available for this dataset")

        # Load headers from Parquet with filtering (using Polars for speed)
        if trace_indices is not None and len(trace_indices) > 0:
            # Convert to list for filtering
            trace_list = trace_indices.tolist() if isinstance(trace_indices, np.ndarray) else list(trace_indices)

            try:
                # Try Parquet filtering for efficiency
                # This loads only relevant row groups instead of entire file
                df = read_parquet(
                    self.headers_path,
                    filters=[('trace_index', 'in', trace_list)]
                )
            except (ValueError, KeyError):
                # Fallback: load all and filter (if trace_index column doesn't exist or filtering fails)
                df = read_parquet(self.headers_path)
                if 'trace_index' in df.columns:
                    df = df[df['trace_index'].isin(trace_list)]
        else:
            # Load all headers (use with caution)
            df = read_parquet(self.headers_path)

        return df

    def get_memory_footprint(self) -> int:
        """
        Get approximate memory footprint of this object.

        Returns:
            Memory usage in bytes

        Note:
            This does not include cached data or the Zarr array itself
            (which is memory-mapped, not loaded).
        """
        footprint = 0

        # Metadata
        footprint += len(str(self.metadata).encode())

        # Ensemble index (if loaded)
        if self._ensemble_index is not None:
            footprint += self._ensemble_index.memory_usage(deep=True).sum()

        # Object overhead (rough estimate)
        footprint += 1024  # Python object overhead

        return footprint

    def __repr__(self) -> str:
        return (f"LazySeismicData(n_samples={self.n_samples}, n_traces={self.n_traces}, "
                f"sample_rate={self.sample_rate}ms, nyquist={self.nyquist_freq:.1f}Hz, "
                f"ensembles={self.get_ensemble_count()}, "
                f"memory={self.get_memory_footprint()/1024/1024:.1f}MB)")

    def __str__(self) -> str:
        return self.__repr__()
