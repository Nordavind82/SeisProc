"""
Shared data management for fork copy-on-write memory sharing.

This module provides a centralized way to pre-load data before forking
worker processes, allowing them to share memory via copy-on-write (COW).

On Linux with fork context:
- Data loaded BEFORE ProcessPoolExecutor creation is inherited by workers
- Workers share read-only pages with parent (COW)
- Only modified pages are duplicated
- Saves ~80-170MB per worker for large datasets

On Windows/macOS with spawn context:
- Workers start fresh and must load data from files
- This module still provides fallback mechanisms

Usage in Coordinator (before fork):
    from utils.parallel_processing.shared_data import (
        set_shared_headers,
        set_shared_ensemble_index,
        clear_shared_data
    )

    # Pre-load before creating ProcessPoolExecutor
    headers_df = read_parquet(headers_path, columns=needed_cols)
    set_shared_headers(headers_df, needed_cols)

    ensemble_df = read_parquet(ensemble_path)
    set_shared_ensemble_index(ensemble_df)

    try:
        with ProcessPoolExecutor(...) as executor:
            # Workers inherit pre-loaded data via fork COW
            ...
    finally:
        clear_shared_data()  # Cleanup

Usage in Worker (after fork):
    from utils.parallel_processing.shared_data import (
        get_shared_headers,
        get_shared_ensemble_index,
        get_shared_ensemble_arrays
    )

    # Access pre-loaded data (zero-copy via COW)
    headers_df, columns = get_shared_headers()
    ensemble_df = get_shared_ensemble_index()
    ensemble_arrays = get_shared_ensemble_arrays()  # Numpy arrays for speed
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict
import logging
import sys

logger = logging.getLogger(__name__)

# ============================================================================
# Module-level globals for fork COW sharing
# These are inherited by child processes when using fork context
# ============================================================================

# Shared headers DataFrame and column list
_SHARED_HEADERS_DF: Optional[pd.DataFrame] = None
_SHARED_HEADERS_COLUMNS: Optional[List[str]] = None

# Shared ensemble index DataFrame
_SHARED_ENSEMBLE_INDEX: Optional[pd.DataFrame] = None

# Pre-converted numpy arrays for fast access during processing
_SHARED_ENSEMBLE_ARRAYS: Optional[Dict[str, np.ndarray]] = None


# ============================================================================
# Headers Management (existing functionality, centralized)
# ============================================================================

def set_shared_headers(headers_df: pd.DataFrame, columns: List[str]) -> None:
    """
    Pre-load headers for fork COW sharing.

    Call this in the coordinator BEFORE creating ProcessPoolExecutor.
    Workers will inherit this data via fork copy-on-write.

    Args:
        headers_df: DataFrame containing header columns
        columns: List of column names that were loaded

    Example:
        headers_df = read_parquet(path, columns=['offset', 'cdp'])
        set_shared_headers(headers_df, ['offset', 'cdp'])
    """
    global _SHARED_HEADERS_DF, _SHARED_HEADERS_COLUMNS

    _SHARED_HEADERS_DF = headers_df
    _SHARED_HEADERS_COLUMNS = columns.copy() if columns else []

    memory_mb = headers_df.memory_usage(deep=True).sum() / (1024 * 1024)
    logger.debug(
        f"Shared headers set: {len(headers_df):,} rows, "
        f"columns={columns}, memory={memory_mb:.1f}MB"
    )


def get_shared_headers() -> Tuple[Optional[pd.DataFrame], Optional[List[str]]]:
    """
    Get pre-loaded shared headers.

    Returns:
        Tuple of (headers_df, columns_list) or (None, None) if not set

    Example:
        headers_df, columns = get_shared_headers()
        if headers_df is not None:
            # Use shared headers (zero-copy via COW)
            offset = headers_df['offset'].values
    """
    return _SHARED_HEADERS_DF, _SHARED_HEADERS_COLUMNS


def has_shared_headers() -> bool:
    """Check if shared headers are available."""
    return _SHARED_HEADERS_DF is not None


def shared_headers_have_columns(required_columns: List[str]) -> bool:
    """
    Check if shared headers contain all required columns.

    Uses case-insensitive matching.

    Args:
        required_columns: List of column names needed

    Returns:
        True if all columns are available (case-insensitive)
    """
    if _SHARED_HEADERS_DF is None or _SHARED_HEADERS_COLUMNS is None:
        return False

    available_lower = {c.lower() for c in _SHARED_HEADERS_COLUMNS}

    for col in required_columns:
        if col not in _SHARED_HEADERS_COLUMNS and col.lower() not in available_lower:
            return False

    return True


# ============================================================================
# Ensemble Index Management (NEW)
# ============================================================================

def set_shared_ensemble_index(ensemble_df: pd.DataFrame) -> None:
    """
    Pre-load ensemble index for fork COW sharing.

    Also converts to numpy arrays for faster access during gather processing.
    The ensemble index maps gather IDs to trace ranges.

    Args:
        ensemble_df: DataFrame with columns like 'start_trace', 'end_trace',
                    optionally 'ensemble_key', 'n_traces'

    Example:
        ensemble_df = read_parquet(ensemble_path)
        set_shared_ensemble_index(ensemble_df)
    """
    global _SHARED_ENSEMBLE_INDEX, _SHARED_ENSEMBLE_ARRAYS

    _SHARED_ENSEMBLE_INDEX = ensemble_df

    # Pre-convert to numpy arrays for O(1) access during processing
    # This avoids DataFrame indexing overhead in hot paths
    _SHARED_ENSEMBLE_ARRAYS = {}

    # Required columns
    if 'start_trace' in ensemble_df.columns:
        _SHARED_ENSEMBLE_ARRAYS['start_trace'] = ensemble_df['start_trace'].to_numpy()

    if 'end_trace' in ensemble_df.columns:
        _SHARED_ENSEMBLE_ARRAYS['end_trace'] = ensemble_df['end_trace'].to_numpy()

    # Optional columns
    if 'ensemble_key' in ensemble_df.columns:
        _SHARED_ENSEMBLE_ARRAYS['ensemble_key'] = ensemble_df['ensemble_key'].to_numpy()

    if 'n_traces' in ensemble_df.columns:
        _SHARED_ENSEMBLE_ARRAYS['n_traces'] = ensemble_df['n_traces'].to_numpy()

    memory_mb = ensemble_df.memory_usage(deep=True).sum() / (1024 * 1024)
    logger.debug(
        f"Shared ensemble index set: {len(ensemble_df):,} gathers, "
        f"memory={memory_mb:.1f}MB"
    )


def get_shared_ensemble_index() -> Optional[pd.DataFrame]:
    """
    Get pre-loaded ensemble index as DataFrame.

    Returns:
        DataFrame with ensemble boundaries, or None if not set

    Example:
        ensemble_df = get_shared_ensemble_index()
        if ensemble_df is not None:
            for idx, row in ensemble_df.iterrows():
                start, end = row['start_trace'], row['end_trace']
    """
    return _SHARED_ENSEMBLE_INDEX


def get_shared_ensemble_arrays() -> Optional[Dict[str, np.ndarray]]:
    """
    Get pre-loaded ensemble index as numpy arrays for fast access.

    Returns arrays for 'start_trace', 'end_trace', and optionally
    'ensemble_key', 'n_traces'.

    Faster than DataFrame access for hot paths.

    Returns:
        Dictionary mapping column names to numpy arrays, or None if not set

    Example:
        arrays = get_shared_ensemble_arrays()
        if arrays is not None:
            starts = arrays['start_trace']
            ends = arrays['end_trace']
            for i in range(len(starts)):
                process_gather(starts[i], ends[i])
    """
    return _SHARED_ENSEMBLE_ARRAYS


def has_shared_ensemble_index() -> bool:
    """Check if shared ensemble index is available."""
    return _SHARED_ENSEMBLE_INDEX is not None


def get_shared_ensemble_bounds(gather_idx: int) -> Optional[Tuple[int, int]]:
    """
    Get trace bounds for a specific gather from shared arrays.

    Fast O(1) access using pre-converted numpy arrays.

    Args:
        gather_idx: Index of gather (0-based)

    Returns:
        Tuple of (start_trace, end_trace) or None if not available
    """
    if _SHARED_ENSEMBLE_ARRAYS is None:
        return None

    starts = _SHARED_ENSEMBLE_ARRAYS.get('start_trace')
    ends = _SHARED_ENSEMBLE_ARRAYS.get('end_trace')

    if starts is None or ends is None:
        return None

    if gather_idx < 0 or gather_idx >= len(starts):
        return None

    return int(starts[gather_idx]), int(ends[gather_idx])


# ============================================================================
# Cleanup
# ============================================================================

def clear_shared_headers() -> None:
    """Clear only shared headers data."""
    global _SHARED_HEADERS_DF, _SHARED_HEADERS_COLUMNS
    _SHARED_HEADERS_DF = None
    _SHARED_HEADERS_COLUMNS = None
    logger.debug("Shared headers cleared")


def clear_shared_ensemble_index() -> None:
    """Clear only shared ensemble index data."""
    global _SHARED_ENSEMBLE_INDEX, _SHARED_ENSEMBLE_ARRAYS
    _SHARED_ENSEMBLE_INDEX = None
    _SHARED_ENSEMBLE_ARRAYS = None
    logger.debug("Shared ensemble index cleared")


def clear_shared_data() -> None:
    """
    Clear all shared data.

    Call this in the coordinator's finally block after processing completes
    to free memory.

    Example:
        try:
            with ProcessPoolExecutor(...) as executor:
                ...
        finally:
            clear_shared_data()
    """
    clear_shared_headers()
    clear_shared_ensemble_index()
    logger.debug("All shared data cleared")


# ============================================================================
# Diagnostics
# ============================================================================

def get_shared_data_memory_usage() -> Dict[str, int]:
    """
    Get memory usage of all shared data in bytes.

    Returns:
        Dictionary with memory breakdown by component
    """
    usage = {
        'headers_df': 0,
        'ensemble_index': 0,
        'ensemble_arrays': 0,
        'total': 0
    }

    if _SHARED_HEADERS_DF is not None:
        usage['headers_df'] = int(_SHARED_HEADERS_DF.memory_usage(deep=True).sum())

    if _SHARED_ENSEMBLE_INDEX is not None:
        usage['ensemble_index'] = int(_SHARED_ENSEMBLE_INDEX.memory_usage(deep=True).sum())

    if _SHARED_ENSEMBLE_ARRAYS is not None:
        for arr in _SHARED_ENSEMBLE_ARRAYS.values():
            usage['ensemble_arrays'] += arr.nbytes

    usage['total'] = sum(v for k, v in usage.items() if k != 'total')

    return usage


def get_shared_data_summary() -> str:
    """
    Get human-readable summary of shared data state.

    Returns:
        Multi-line string describing current state
    """
    lines = ["Shared Data Summary:"]

    if _SHARED_HEADERS_DF is not None:
        mem = _SHARED_HEADERS_DF.memory_usage(deep=True).sum() / (1024 * 1024)
        lines.append(
            f"  Headers: {len(_SHARED_HEADERS_DF):,} rows, "
            f"columns={_SHARED_HEADERS_COLUMNS}, {mem:.1f}MB"
        )
    else:
        lines.append("  Headers: Not loaded")

    if _SHARED_ENSEMBLE_INDEX is not None:
        mem = _SHARED_ENSEMBLE_INDEX.memory_usage(deep=True).sum() / (1024 * 1024)
        lines.append(f"  Ensemble Index: {len(_SHARED_ENSEMBLE_INDEX):,} gathers, {mem:.1f}MB")
    else:
        lines.append("  Ensemble Index: Not loaded")

    if _SHARED_ENSEMBLE_ARRAYS is not None:
        arr_mem = sum(arr.nbytes for arr in _SHARED_ENSEMBLE_ARRAYS.values()) / (1024 * 1024)
        lines.append(f"  Ensemble Arrays: {list(_SHARED_ENSEMBLE_ARRAYS.keys())}, {arr_mem:.1f}MB")

    usage = get_shared_data_memory_usage()
    lines.append(f"  Total Memory: {usage['total'] / (1024 * 1024):.1f}MB")

    return "\n".join(lines)


def is_fork_context() -> bool:
    """
    Check if we're likely using fork context (Linux).

    Fork context enables COW memory sharing for pre-loaded data.

    Returns:
        True if on Linux (fork context expected)
    """
    return sys.platform == 'linux'


# ============================================================================
# Backward Compatibility
# ============================================================================

# These aliases maintain compatibility with existing worker.py code
# that imports from worker module. Can be removed after migration.

def set_shared_headers_compat(headers_df: pd.DataFrame, columns: List[str]) -> None:
    """Backward compatible alias for set_shared_headers."""
    set_shared_headers(headers_df, columns)


def get_shared_headers_compat() -> Tuple[Optional[pd.DataFrame], Optional[List[str]]]:
    """Backward compatible alias for get_shared_headers."""
    return get_shared_headers()


def clear_shared_headers_compat() -> None:
    """Backward compatible alias for clear_shared_headers."""
    clear_shared_headers()
