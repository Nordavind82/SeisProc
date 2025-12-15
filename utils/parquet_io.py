"""
Parquet I/O abstraction layer with Polars acceleration.

Provides unified API for reading/writing Parquet files with:
- Polars for speed (6x faster than Pandas for large files)
- Automatic fallback to Pandas if Polars unavailable
- Optional conversion to Pandas DataFrames for compatibility

Usage:
    from utils.parquet_io import read_parquet, read_parquet_schema

    # Fast read with Polars, returns Pandas DataFrame
    df = read_parquet('headers.parquet', columns=['offset', 'cdp'])

    # Get schema without loading data
    columns = read_parquet_schema('headers.parquet')
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Union, Any, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Try to import Polars
try:
    import polars as pl
    POLARS_AVAILABLE = True
    _POLARS_VERSION = pl.__version__
    logger.debug(f"Polars {_POLARS_VERSION} available for accelerated Parquet I/O")
except ImportError:
    POLARS_AVAILABLE = False
    _POLARS_VERSION = None
    pl = None
    logger.info("Polars not available, using Pandas for Parquet I/O")

# PyArrow for schema reading (usually available with Pandas)
try:
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    pq = None


def read_parquet(
    path: Union[str, Path],
    columns: Optional[List[str]] = None,
    use_polars: bool = True,
    return_pandas: bool = True,
    **kwargs
) -> Union[pd.DataFrame, 'pl.DataFrame']:
    """
    Read Parquet file with Polars (if available) or Pandas fallback.

    Polars provides ~6x faster read times for large files and uses
    ~50% less memory than Pandas.

    Args:
        path: Path to parquet file
        columns: Optional list of columns to load (reduces memory/time)
        use_polars: Whether to use Polars if available (default True)
        return_pandas: Convert result to Pandas DataFrame (default True)
        **kwargs: Additional arguments passed to underlying reader

    Returns:
        DataFrame (Pandas by default, or Polars if return_pandas=False)

    Example:
        # Read specific columns (fastest)
        df = read_parquet('headers.parquet', columns=['offset', 'cdp'])

        # Read all columns
        df = read_parquet('headers.parquet')

        # Get Polars DataFrame directly (for chained operations)
        pl_df = read_parquet('data.parquet', return_pandas=False)
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    # Check if filters are requested - Polars doesn't support PyArrow filter syntax
    has_filters = 'filters' in kwargs and kwargs['filters']

    # Try Polars first (faster) - but skip if filters requested
    if POLARS_AVAILABLE and use_polars and not has_filters:
        try:
            # Filter kwargs for Polars compatibility
            polars_kwargs = {k: v for k, v in kwargs.items()
                           if k in ('n_rows', 'row_count_name', 'row_count_offset',
                                   'parallel', 'use_statistics', 'hive_partitioning',
                                   'rechunk', 'low_memory')}

            df = pl.read_parquet(path, columns=columns, **polars_kwargs)

            if return_pandas:
                return df.to_pandas()
            return df

        except Exception as e:
            logger.warning(f"Polars read failed for {path}, falling back to Pandas: {e}")

    # Pandas fallback
    pandas_kwargs = {k: v for k, v in kwargs.items()
                    if k in ('engine', 'use_nullable_dtypes', 'dtype_backend',
                            'filesystem', 'filters', 'storage_options')}

    return pd.read_parquet(path, columns=columns, **pandas_kwargs)


def read_parquet_lazy(
    path: Union[str, Path],
    columns: Optional[List[str]] = None,
) -> 'pl.LazyFrame':
    """
    Create lazy Polars scanner for Parquet file.

    Lazy evaluation allows query optimization and reduces memory usage
    for complex operations. Only reads data when .collect() is called.

    Args:
        path: Path to parquet file
        columns: Optional list of columns to load

    Returns:
        Polars LazyFrame for chained operations

    Raises:
        ImportError: If Polars is not available

    Example:
        # Lazy operations (optimized query plan)
        result = (
            read_parquet_lazy('headers.parquet')
            .filter(pl.col('offset') > 1000)
            .select(['offset', 'cdp'])
            .collect()  # Execute here
        )
    """
    if not POLARS_AVAILABLE:
        raise ImportError(
            "Polars required for lazy reading. "
            "Install with: pip install polars"
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    return pl.scan_parquet(path, columns=columns)


def read_parquet_schema(path: Union[str, Path]) -> List[str]:
    """
    Get column names from Parquet file without loading data.

    Fast operation that only reads file metadata.

    Args:
        path: Path to parquet file

    Returns:
        List of column names

    Example:
        columns = read_parquet_schema('headers.parquet')
        # ['trace_index', 'offset', 'cdp', 'inline', 'xline', ...]
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    # Try Polars first (fastest)
    if POLARS_AVAILABLE:
        try:
            schema = pl.read_parquet_schema(path)
            return list(schema.keys())
        except Exception:
            pass

    # PyArrow fallback
    if PYARROW_AVAILABLE:
        try:
            schema = pq.read_schema(path)
            return schema.names
        except Exception:
            pass

    # Last resort: read with Pandas and get columns
    df = pd.read_parquet(path, columns=[])
    return list(df.columns)


def read_parquet_schema_with_types(
    path: Union[str, Path]
) -> Dict[str, str]:
    """
    Get column names and data types from Parquet file.

    Args:
        path: Path to parquet file

    Returns:
        Dictionary mapping column names to type strings

    Example:
        schema = read_parquet_schema_with_types('headers.parquet')
        # {'trace_index': 'Int64', 'offset': 'Float64', ...}
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    if POLARS_AVAILABLE:
        try:
            schema = pl.read_parquet_schema(path)
            return {name: str(dtype) for name, dtype in schema.items()}
        except Exception:
            pass

    if PYARROW_AVAILABLE:
        try:
            schema = pq.read_schema(path)
            return {name: str(schema.field(name).type) for name in schema.names}
        except Exception:
            pass

    # Pandas fallback - read small sample
    df = pd.read_parquet(path, columns=None)
    return {col: str(df[col].dtype) for col in df.columns}


def write_parquet(
    df: Union[pd.DataFrame, 'pl.DataFrame'],
    path: Union[str, Path],
    compression: str = 'zstd',
    use_polars: bool = True,
    **kwargs
) -> None:
    """
    Write DataFrame to Parquet file.

    Args:
        df: Pandas or Polars DataFrame
        path: Output path
        compression: Compression algorithm ('zstd', 'lz4', 'snappy', 'gzip', None)
        use_polars: Use Polars for writing if available
        **kwargs: Additional arguments for writer

    Example:
        write_parquet(df, 'output.parquet', compression='lz4')
    """
    path = Path(path)

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # If already Polars DataFrame
    if POLARS_AVAILABLE and isinstance(df, pl.DataFrame):
        df.write_parquet(path, compression=compression, **kwargs)
        return

    # Try Polars for Pandas DataFrame
    if POLARS_AVAILABLE and use_polars and isinstance(df, pd.DataFrame):
        try:
            pl_df = pl.from_pandas(df)
            pl_df.write_parquet(path, compression=compression)
            return
        except Exception as e:
            logger.warning(f"Polars write failed, using Pandas: {e}")

    # Pandas write
    if isinstance(df, pd.DataFrame):
        df.to_parquet(path, compression=compression, **kwargs)
    else:
        raise TypeError(f"Expected DataFrame, got {type(df)}")


def get_parquet_row_count(path: Union[str, Path]) -> int:
    """
    Get row count from Parquet file without loading data.

    Args:
        path: Path to parquet file

    Returns:
        Number of rows in file
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    if PYARROW_AVAILABLE:
        try:
            parquet_file = pq.ParquetFile(path)
            return parquet_file.metadata.num_rows
        except Exception:
            pass

    if POLARS_AVAILABLE:
        try:
            return pl.scan_parquet(path).select(pl.count()).collect().item()
        except Exception:
            pass

    # Fallback: read single column and count
    df = pd.read_parquet(path, columns=[pd.read_parquet(path, columns=[]).columns[0]])
    return len(df)


def find_columns_case_insensitive(
    path: Union[str, Path],
    target_columns: List[str]
) -> Dict[str, Optional[str]]:
    """
    Find columns in Parquet file with case-insensitive matching.

    Useful for handling header files with inconsistent column naming
    (e.g., 'offset', 'OFFSET', 'Offset').

    Args:
        path: Path to parquet file
        target_columns: List of column names to find

    Returns:
        Dictionary mapping target names to actual column names (or None if not found)

    Example:
        mapping = find_columns_case_insensitive(
            'headers.parquet',
            ['offset', 'cdp', 'inline']
        )
        # {'offset': 'OFFSET', 'cdp': 'CDP', 'inline': None}
    """
    available = read_parquet_schema(path)
    available_lower = {col.lower(): col for col in available}

    result = {}
    for target in target_columns:
        if target in available:
            result[target] = target
        elif target.lower() in available_lower:
            result[target] = available_lower[target.lower()]
        else:
            result[target] = None

    return result


# Convenience function for common pattern
def read_parquet_columns_matched(
    path: Union[str, Path],
    columns: List[str],
    case_insensitive: bool = True
) -> pd.DataFrame:
    """
    Read Parquet file with case-insensitive column matching.

    Automatically finds columns regardless of case and returns
    DataFrame with requested column names.

    Args:
        path: Path to parquet file
        columns: List of desired column names
        case_insensitive: Whether to match columns case-insensitively

    Returns:
        DataFrame with columns renamed to requested names

    Raises:
        ValueError: If required columns not found

    Example:
        # Works regardless of whether file has 'offset' or 'OFFSET'
        df = read_parquet_columns_matched(
            'headers.parquet',
            ['offset', 'cdp']
        )
    """
    if case_insensitive:
        mapping = find_columns_case_insensitive(path, columns)
        missing = [col for col, found in mapping.items() if found is None]

        if missing:
            available = read_parquet_schema(path)
            raise ValueError(
                f"Columns not found: {missing}. "
                f"Available columns: {available[:20]}..."
            )

        # Read actual columns
        actual_columns = [mapping[col] for col in columns]
        df = read_parquet(path, columns=actual_columns)

        # Rename to requested names
        rename_map = {mapping[col]: col for col in columns if mapping[col] != col}
        if rename_map:
            df = df.rename(columns=rename_map)

        return df

    else:
        return read_parquet(path, columns=columns)
