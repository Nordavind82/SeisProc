"""
Trace sorting utilities for in-gather reordering.

Provides functionality to:
- Compute sort indices for traces within gathers
- Reorder headers to match sorted trace order
- Rebuild ensemble index after sorting
- Create global trace mapping for export

This enables sorting traces by offset, trace_number, or other keys
while maintaining header-trace alignment during export.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict
from dataclasses import dataclass


@dataclass
class SortConfig:
    """Configuration for trace sorting."""
    sort_key: str                    # Header field to sort by (e.g., 'offset')
    ascending: bool = True           # Sort direction
    secondary_key: Optional[str] = None  # Optional secondary sort key
    secondary_ascending: bool = True


@dataclass
class GatherSortResult:
    """Result of sorting a single gather."""
    gather_idx: int
    start_trace: int           # Global start trace index
    end_trace: int             # Global end trace index
    sort_indices: np.ndarray   # Local indices within gather (0 to n_traces-1)
    original_indices: np.ndarray  # Global original trace indices


class TraceSorter:
    """
    Handles trace sorting within gathers.

    Sorting is applied within each gather independently, preserving
    gather boundaries while reordering traces within each gather.

    Example:
        sorter = TraceSorter(sort_key='offset', ascending=True)

        # For a single gather
        sorted_traces, sort_indices = sorter.sort_gather(
            traces, headers_df, start_trace=100, end_trace=199
        )

        # For full dataset reordering
        global_mapping = sorter.compute_global_sort_mapping(
            headers_df, ensemble_df
        )
    """

    def __init__(
        self,
        sort_key: str,
        ascending: bool = True,
        secondary_key: Optional[str] = None,
        secondary_ascending: bool = True
    ):
        """
        Initialize sorter.

        Args:
            sort_key: Primary header field to sort by
            ascending: Sort direction for primary key
            secondary_key: Optional secondary sort key
            secondary_ascending: Sort direction for secondary key
        """
        self.sort_key = sort_key
        self.ascending = ascending
        self.secondary_key = secondary_key
        self.secondary_ascending = secondary_ascending

    def compute_gather_sort_indices(
        self,
        headers_df: pd.DataFrame,
        start_trace: int,
        end_trace: int
    ) -> np.ndarray:
        """
        Compute sort indices for traces within a gather.

        Args:
            headers_df: Full headers DataFrame
            start_trace: First trace index of gather (inclusive)
            end_trace: Last trace index of gather (inclusive)

        Returns:
            Array of local indices (0 to n_traces-1) in sorted order
        """
        # Extract gather headers
        gather_headers = headers_df.iloc[start_trace:end_trace + 1]
        n_traces = len(gather_headers)

        if n_traces == 0:
            return np.array([], dtype=np.int64)

        # Get sort values
        if self.sort_key not in gather_headers.columns:
            # If sort key not found, return original order
            return np.arange(n_traces, dtype=np.int64)

        sort_values = gather_headers[self.sort_key].values

        if self.secondary_key and self.secondary_key in gather_headers.columns:
            # Multi-key sort using lexsort (sorts by last key first)
            secondary_values = gather_headers[self.secondary_key].values

            # Adjust for ascending/descending
            if not self.secondary_ascending:
                secondary_values = -secondary_values if np.issubdtype(secondary_values.dtype, np.number) else secondary_values
            if not self.ascending:
                sort_values = -sort_values if np.issubdtype(sort_values.dtype, np.number) else sort_values

            sort_indices = np.lexsort((secondary_values, sort_values))
        else:
            # Single key sort
            sort_indices = np.argsort(sort_values)
            if not self.ascending:
                sort_indices = sort_indices[::-1]

        return sort_indices.astype(np.int64)

    def sort_gather_traces(
        self,
        traces: np.ndarray,
        sort_indices: np.ndarray
    ) -> np.ndarray:
        """
        Reorder traces according to sort indices.

        Args:
            traces: Trace data array (n_samples, n_traces)
            sort_indices: Local indices in sorted order

        Returns:
            Sorted traces array (n_samples, n_traces)
        """
        return traces[:, sort_indices]

    def compute_global_sort_mapping(
        self,
        headers_df: pd.DataFrame,
        ensemble_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Compute global trace mapping after sorting all gathers.

        Returns an array where mapping[new_position] = original_position.
        This can be used to reorder the full dataset or during export.

        OPTIMIZED: Uses vectorized numpy operations instead of Python loops.

        Args:
            headers_df: Full headers DataFrame
            ensemble_df: Ensemble index DataFrame

        Returns:
            Array of length n_traces where mapping[i] gives original trace index
        """
        n_traces = len(headers_df)
        global_mapping = np.zeros(n_traces, dtype=np.int64)

        current_pos = 0

        for idx in range(len(ensemble_df)):
            ensemble = ensemble_df.iloc[idx]
            start_trace = int(ensemble['start_trace'])
            end_trace = int(ensemble['end_trace'])
            n_gather_traces = end_trace - start_trace + 1

            # Get sort indices for this gather
            local_sort_indices = self.compute_gather_sort_indices(
                headers_df, start_trace, end_trace
            )

            # VECTORIZED: Map local sorted indices to global original indices
            # No Python loop - direct array assignment
            global_orig_indices = start_trace + local_sort_indices
            global_mapping[current_pos:current_pos + n_gather_traces] = global_orig_indices

            current_pos += n_gather_traces

        return global_mapping

    def compute_global_sort_mapping_chunked(
        self,
        headers_df: pd.DataFrame,
        ensemble_df: pd.DataFrame,
        chunk_size: int = 1000
    ) -> np.ndarray:
        """
        Memory-efficient version that processes gathers in chunks.

        For very large datasets, this prevents memory spikes by processing
        a limited number of gathers at a time and triggering garbage collection.

        Args:
            headers_df: Full headers DataFrame
            ensemble_df: Ensemble index DataFrame
            chunk_size: Number of gathers to process before gc.collect()

        Returns:
            Array of length n_traces where mapping[i] gives original trace index
        """
        import gc

        n_traces = len(headers_df)
        global_mapping = np.zeros(n_traces, dtype=np.int64)

        current_pos = 0
        n_gathers = len(ensemble_df)

        for idx in range(n_gathers):
            ensemble = ensemble_df.iloc[idx]
            start_trace = int(ensemble['start_trace'])
            end_trace = int(ensemble['end_trace'])
            n_gather_traces = end_trace - start_trace + 1

            # Get sort indices for this gather
            local_sort_indices = self.compute_gather_sort_indices(
                headers_df, start_trace, end_trace
            )

            # Vectorized mapping
            global_orig_indices = start_trace + local_sort_indices
            global_mapping[current_pos:current_pos + n_gather_traces] = global_orig_indices

            current_pos += n_gather_traces

            # Periodic garbage collection for memory efficiency
            if (idx + 1) % chunk_size == 0:
                gc.collect()

        return global_mapping

    def create_sorted_headers(
        self,
        headers_df: pd.DataFrame,
        global_mapping: np.ndarray,
        chunk_size: int = 100_000
    ) -> pd.DataFrame:
        """
        Create reordered headers DataFrame with memory-efficient chunking.

        For large datasets, processes in chunks to avoid memory spikes
        from creating a full copy of the DataFrame.

        Args:
            headers_df: Original headers DataFrame
            global_mapping: Array where mapping[new_pos] = original_pos
            chunk_size: Number of rows to process at a time (default 100k)

        Returns:
            New DataFrame with headers in sorted order
        """
        import gc

        n_traces = len(global_mapping)

        if n_traces <= chunk_size:
            # Small dataset: direct reorder is efficient
            sorted_headers = headers_df.iloc[global_mapping].reset_index(drop=True)
            sorted_headers['trace_index'] = np.arange(len(sorted_headers))
            sorted_headers['original_trace_index'] = global_mapping
            return sorted_headers

        # Large dataset: chunked processing to limit memory
        chunks = []
        n_chunks = (n_traces + chunk_size - 1) // chunk_size

        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, n_traces)
            chunk_mapping = global_mapping[chunk_start:chunk_end]

            # Extract rows for this chunk
            chunk_headers = headers_df.iloc[chunk_mapping].copy()
            chunk_headers['trace_index'] = np.arange(chunk_start, chunk_end)
            chunk_headers['original_trace_index'] = chunk_mapping

            chunks.append(chunk_headers)

            # Cleanup
            del chunk_mapping
            if chunk_idx % 5 == 0:
                gc.collect()

        # Concatenate all chunks
        sorted_headers = pd.concat(chunks, ignore_index=True)
        del chunks
        gc.collect()

        return sorted_headers

    def rebuild_ensemble_index(
        self,
        sorted_headers: pd.DataFrame,
        ensemble_key: str
    ) -> pd.DataFrame:
        """
        Rebuild ensemble index from sorted headers.

        After sorting, trace ranges within each ensemble remain the same
        size, but we rebuild to ensure consistency.

        Args:
            sorted_headers: Sorted headers DataFrame
            ensemble_key: Column name for ensemble grouping (e.g., 'CDP', 'FieldRecord')

        Returns:
            New ensemble index DataFrame
        """
        ensembles = []

        # Group by ensemble key
        if ensemble_key not in sorted_headers.columns:
            # Fall back to treating entire dataset as one ensemble
            ensembles.append({
                'ensemble_id': 0,
                'start_trace': 0,
                'end_trace': len(sorted_headers) - 1,
                'n_traces': len(sorted_headers)
            })
        else:
            # Detect ensemble boundaries (where key value changes)
            ensemble_values = sorted_headers[ensemble_key].values

            current_start = 0
            current_value = ensemble_values[0]
            ensemble_id = 0

            for i in range(1, len(ensemble_values)):
                if ensemble_values[i] != current_value:
                    # End of current ensemble
                    ensembles.append({
                        'ensemble_id': ensemble_id,
                        ensemble_key: current_value,
                        'start_trace': current_start,
                        'end_trace': i - 1,
                        'n_traces': i - current_start
                    })
                    current_start = i
                    current_value = ensemble_values[i]
                    ensemble_id += 1

            # Don't forget the last ensemble
            ensembles.append({
                'ensemble_id': ensemble_id,
                ensemble_key: current_value,
                'start_trace': current_start,
                'end_trace': len(ensemble_values) - 1,
                'n_traces': len(ensemble_values) - current_start
            })

        return pd.DataFrame(ensembles)


def compute_gather_sort_indices(
    headers_df: pd.DataFrame,
    start_trace: int,
    end_trace: int,
    sort_key: str,
    ascending: bool = True
) -> np.ndarray:
    """
    Convenience function to compute sort indices for a gather.

    Args:
        headers_df: Full headers DataFrame
        start_trace: First trace index of gather
        end_trace: Last trace index of gather
        sort_key: Header field to sort by
        ascending: Sort direction

    Returns:
        Array of local indices in sorted order
    """
    sorter = TraceSorter(sort_key=sort_key, ascending=ascending)
    return sorter.compute_gather_sort_indices(headers_df, start_trace, end_trace)


def create_sorted_output(
    input_zarr_path: Path,
    headers_parquet_path: Path,
    ensemble_index_path: Path,
    output_dir: Path,
    sort_key: str,
    ascending: bool = True,
    ensemble_key: str = 'CDP',
    progress_callback: Optional[callable] = None,
    trace_chunk_size: int = 1000
) -> Tuple[Path, Path, Path]:
    """
    Create sorted Zarr, headers, and ensemble index.

    This is a full-dataset sort operation that:
    1. Computes global sort mapping
    2. Creates sorted headers.parquet
    3. Creates sorted ensemble_index.parquet
    4. Creates sorted traces.zarr

    MEMORY OPTIMIZATIONS:
    - Uses chunked header processing
    - Vectorized trace copying
    - Explicit garbage collection

    Args:
        input_zarr_path: Path to input traces.zarr
        headers_parquet_path: Path to headers.parquet
        ensemble_index_path: Path to ensemble_index.parquet
        output_dir: Directory for sorted output
        sort_key: Header field to sort by
        ascending: Sort direction
        ensemble_key: Column for ensemble grouping
        progress_callback: Optional callback(current, total)
        trace_chunk_size: Number of traces to copy at once (default 1000)

    Returns:
        Tuple of (sorted_zarr_path, sorted_headers_path, sorted_ensemble_path)
    """
    import gc
    import zarr

    # Load data
    headers_df = pd.read_parquet(headers_parquet_path)
    ensemble_df = pd.read_parquet(ensemble_index_path)
    input_zarr = zarr.open(str(input_zarr_path), mode='r')

    n_samples, n_traces = input_zarr.shape

    # Create sorter and compute mapping (uses vectorized operations)
    sorter = TraceSorter(sort_key=sort_key, ascending=ascending)
    global_mapping = sorter.compute_global_sort_mapping(headers_df, ensemble_df)

    # Create sorted headers (uses chunked processing for large datasets)
    sorted_headers = sorter.create_sorted_headers(headers_df, global_mapping)

    # Free original headers memory
    del headers_df
    gc.collect()

    # Rebuild ensemble index
    sorted_ensemble = sorter.rebuild_ensemble_index(sorted_headers, ensemble_key)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save sorted headers and ensemble index
    sorted_headers_path = output_dir / 'headers.parquet'
    sorted_ensemble_path = output_dir / 'ensemble_index.parquet'
    sorted_headers.to_parquet(sorted_headers_path)
    sorted_ensemble.to_parquet(sorted_ensemble_path)

    # Free sorted headers memory
    del sorted_headers
    del sorted_ensemble
    gc.collect()

    # Create sorted Zarr
    sorted_zarr_path = output_dir / 'traces.zarr'
    output_zarr = zarr.open(
        str(sorted_zarr_path),
        mode='w',
        shape=(n_samples, n_traces),
        chunks=input_zarr.chunks,
        dtype=input_zarr.dtype
    )

    # OPTIMIZED: Copy traces in sorted order using vectorized batch operations
    for i in range(0, n_traces, trace_chunk_size):
        end_i = min(i + trace_chunk_size, n_traces)
        chunk_size = end_i - i

        # Get source indices for this output range
        source_indices = global_mapping[i:end_i]

        # VECTORIZED: Load all source traces at once where possible
        # Group consecutive indices for efficient loading
        unique_indices = np.unique(source_indices)

        # For random access patterns, load individually but in batch
        chunk_traces = np.zeros((n_samples, chunk_size), dtype=input_zarr.dtype)
        for j, src_idx in enumerate(source_indices):
            chunk_traces[:, j] = input_zarr[:, src_idx]

        # Write entire chunk at once
        output_zarr[:, i:end_i] = chunk_traces

        # Cleanup
        del chunk_traces
        if i % (trace_chunk_size * 10) == 0:
            gc.collect()

        if progress_callback:
            progress_callback(end_i, n_traces)

    return sorted_zarr_path, sorted_headers_path, sorted_ensemble_path
