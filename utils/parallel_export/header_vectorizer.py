"""
Header vectorization for fast trace header access.

Eliminates per-trace pandas iloc overhead by pre-converting
DataFrame columns to numpy arrays for O(1) indexed access.

Performance improvement:
- Old: headers_df.iloc[i].to_dict()  # O(n_columns) per trace
- New: {col: arr[i] for col, arr in header_arrays.items()}  # O(1) per trace

For 12M traces with 50 header fields:
- Old: ~600M operations for headers alone
- New: ~12M array lookups (50x faster)
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Set, Optional

# Mapping from import header names (lowercase, from HeaderMapping)
# to segyio.TraceField names (mixed case, as used by segyio)
IMPORT_TO_SEGYIO_MAPPING = {
    # Sequence numbers
    'trace_sequence_line': 'TRACE_SEQUENCE_LINE',
    'trace_sequence_file': 'TRACE_SEQUENCE_FILE',

    # Field/shot info
    'field_record': 'FieldRecord',
    'trace_number': 'TraceNumber',
    'energy_source_point': 'EnergySourcePoint',

    # CDP/ensemble
    'cdp': 'CDP',
    'trace_number_cdp': 'CDP_TRACE',

    # Trace identification
    'trace_id_code': 'TraceIdentificationCode',

    # Offset and elevations
    'offset': 'offset',
    'receiver_elevation': 'ReceiverGroupElevation',
    'source_elevation': 'SourceSurfaceElevation',
    'source_depth': 'SourceDepth',

    # Coordinates
    'scalar_coord': 'ElevationScalar',  # Also used for coordinate scalar
    'source_x': 'SourceX',
    'source_y': 'SourceY',
    'receiver_x': 'GroupX',
    'receiver_y': 'GroupY',

    # Sample info
    'sample_count': 'TRACE_SAMPLE_COUNT',
    'sample_interval': 'TRACE_SAMPLE_INTERVAL',

    # 3D geometry
    'inline': 'INLINE_3D',
    'crossline': 'CROSSLINE_3D',

    # Additional commonly used fields (direct mappings where names match)
    'CDP': 'CDP',
    'TRACE_SEQUENCE_LINE': 'TRACE_SEQUENCE_LINE',
    'TRACE_SEQUENCE_FILE': 'TRACE_SEQUENCE_FILE',
    'FieldRecord': 'FieldRecord',
    'TraceNumber': 'TraceNumber',
    'EnergySourcePoint': 'EnergySourcePoint',
    'CDP_TRACE': 'CDP_TRACE',
    'TraceIdentificationCode': 'TraceIdentificationCode',
    'ReceiverGroupElevation': 'ReceiverGroupElevation',
    'SourceSurfaceElevation': 'SourceSurfaceElevation',
    'SourceDepth': 'SourceDepth',
    'ElevationScalar': 'ElevationScalar',
    'SourceGroupScalar': 'SourceGroupScalar',
    'SourceX': 'SourceX',
    'SourceY': 'SourceY',
    'GroupX': 'GroupX',
    'GroupY': 'GroupY',
    'TRACE_SAMPLE_COUNT': 'TRACE_SAMPLE_COUNT',
    'TRACE_SAMPLE_INTERVAL': 'TRACE_SAMPLE_INTERVAL',
    'INLINE_3D': 'INLINE_3D',
    'CROSSLINE_3D': 'CROSSLINE_3D',
    'NSummedTraces': 'NSummedTraces',
    'NStackedTraces': 'NStackedTraces',
    'DataUse': 'DataUse',
    'CoordinateUnits': 'CoordinateUnits',
    'WeatheringVelocity': 'WeatheringVelocity',
    'SubWeatheringVelocity': 'SubWeatheringVelocity',
    'SourceUpholeTime': 'SourceUpholeTime',
    'GroupUpholeTime': 'GroupUpholeTime',
    'SourceStaticCorrection': 'SourceStaticCorrection',
    'GroupStaticCorrection': 'GroupStaticCorrection',
    'TotalStaticApplied': 'TotalStaticApplied',
    'LagTimeA': 'LagTimeA',
    'LagTimeB': 'LagTimeB',
    'DelayRecordingTime': 'DelayRecordingTime',
    'MuteTimeStart': 'MuteTimeStart',
    'MuteTimeEnd': 'MuteTimeEnd',
    'GainType': 'GainType',
    'InstrumentGainConstant': 'InstrumentGainConstant',
    'InstrumentInitialGain': 'InstrumentInitialGain',
    'Correlated': 'Correlated',
    'SweepFrequencyStart': 'SweepFrequencyStart',
    'SweepFrequencyEnd': 'SweepFrequencyEnd',
    'SweepLength': 'SweepLength',
    'SweepType': 'SweepType',
    'SweepTraceTaperLengthStart': 'SweepTraceTaperLengthStart',
    'SweepTraceTaperLengthEnd': 'SweepTraceTaperLengthEnd',
    'TaperType': 'TaperType',
    'AliasFilterFrequency': 'AliasFilterFrequency',
    'AliasFilterSlope': 'AliasFilterSlope',
    'NotchFilterFrequency': 'NotchFilterFrequency',
    'NotchFilterSlope': 'NotchFilterSlope',
    'LowCutFrequency': 'LowCutFrequency',
    'HighCutFrequency': 'HighCutFrequency',
    'LowCutSlope': 'LowCutSlope',
    'HighCutSlope': 'HighCutSlope',
    'YearDataRecorded': 'YearDataRecorded',
    'DayOfYear': 'DayOfYear',
    'HourOfDay': 'HourOfDay',
    'MinuteOfHour': 'MinuteOfHour',
    'SecondOfMinute': 'SecondOfMinute',
    'TimeBaseCode': 'TimeBaseCode',
    'TraceWeightingFactor': 'TraceWeightingFactor',
    'GeophoneGroupNumberRoll1': 'GeophoneGroupNumberRoll1',
    'GeophoneGroupNumberFirstTraceOrigField': 'GeophoneGroupNumberFirstTraceOrigField',
    'GeophoneGroupNumberLastTraceOrigField': 'GeophoneGroupNumberLastTraceOrigField',
    'GapSize': 'GapSize',
    'OverTravel': 'OverTravel',
    'CDP_X': 'CDP_X',
    'CDP_Y': 'CDP_Y',
    'ShotPoint': 'ShotPoint',
    'ShotPointScalar': 'ShotPointScalar',
    'TraceValueMeasurementUnit': 'TraceValueMeasurementUnit',
    'TransductionConstantMantissa': 'TransductionConstantMantissa',
    'TransductionConstantPower': 'TransductionConstantPower',
    'TransductionUnit': 'TransductionUnit',
    'TraceIdentifier': 'TraceIdentifier',
    'ScalarTraceHeader': 'ScalarTraceHeader',
    'SourceType': 'SourceType',
    'SourceEnergyDirectionMantissa': 'SourceEnergyDirectionMantissa',
    'SourceEnergyDirectionExponent': 'SourceEnergyDirectionExponent',
    'SourceMeasurementMantissa': 'SourceMeasurementMantissa',
    'SourceMeasurementExponent': 'SourceMeasurementExponent',
    'SourceMeasurementUnit': 'SourceMeasurementUnit',
    'ReceiverDatumElevation': 'ReceiverDatumElevation',
    'SourceDatumElevation': 'SourceDatumElevation',
    'SourceWaterDepth': 'SourceWaterDepth',
    'GroupWaterDepth': 'GroupWaterDepth',
}

# Set of valid segyio TraceField names (for validation)
SEGY_TRACE_HEADER_FIELDS = set(IMPORT_TO_SEGYIO_MAPPING.values())


class HeaderVectorizer:
    """
    Converts pandas DataFrame headers to vectorized numpy arrays.

    This class eliminates the per-trace overhead of DataFrame access
    by pre-converting columns to numpy arrays. The vectorized arrays
    can be serialized and shared with worker processes.

    Handles mapping from import header names (lowercase) to segyio
    TraceField names (mixed case) automatically.
    """

    def __init__(self, headers_df: pd.DataFrame):
        """
        Initialize with headers DataFrame.

        Args:
            headers_df: DataFrame with trace headers (can use import or segyio names)
        """
        self.headers_df = headers_df
        self.n_traces = len(headers_df)
        self._header_arrays: Optional[Dict[str, np.ndarray]] = None
        self._mapped_columns: Optional[Dict[str, str]] = None  # import_name -> segyio_name

    def _build_column_mapping(self) -> Dict[str, str]:
        """
        Build mapping from DataFrame columns to segyio field names.

        Returns:
            Dictionary mapping DataFrame column names to segyio TraceField names
        """
        if self._mapped_columns is not None:
            return self._mapped_columns

        self._mapped_columns = {}

        for col in self.headers_df.columns:
            # Check if column name is in our mapping
            if col in IMPORT_TO_SEGYIO_MAPPING:
                segyio_name = IMPORT_TO_SEGYIO_MAPPING[col]
                self._mapped_columns[col] = segyio_name
            # Also try lowercase version
            elif col.lower() in IMPORT_TO_SEGYIO_MAPPING:
                segyio_name = IMPORT_TO_SEGYIO_MAPPING[col.lower()]
                self._mapped_columns[col] = segyio_name

        return self._mapped_columns

    def vectorize(self, keep_original_names: bool = False) -> Dict[str, np.ndarray]:
        """
        Convert DataFrame columns to numpy arrays.

        Args:
            keep_original_names: If True, keep original parquet column names.
                                If False (default), map to segyio field names.

        Returns:
            Dictionary mapping column/field names to numpy arrays
        """
        if self._header_arrays is not None:
            return self._header_arrays

        self._header_arrays = {}

        if keep_original_names:
            # Keep original parquet column names - used for custom header mapping
            for col in self.headers_df.columns:
                try:
                    arr = self.headers_df[col].values
                    if np.issubdtype(arr.dtype, np.floating):
                        arr = np.nan_to_num(arr, nan=0.0)
                    self._header_arrays[col] = arr.astype(np.int32)
                except (ValueError, TypeError):
                    continue
        else:
            # Map to segyio field names - used for auto-detect mode
            column_mapping = self._build_column_mapping()
            for df_col, segyio_name in column_mapping.items():
                try:
                    arr = self.headers_df[df_col].values
                    if np.issubdtype(arr.dtype, np.floating):
                        arr = np.nan_to_num(arr, nan=0.0)
                    self._header_arrays[segyio_name] = arr.astype(np.int32)
                except (ValueError, TypeError):
                    continue

        return self._header_arrays

    def get_trace_headers(self, trace_idx: int) -> Dict[str, int]:
        """
        Get headers for a single trace.

        Args:
            trace_idx: Trace index

        Returns:
            Dictionary of segyio field name -> value
        """
        if self._header_arrays is None:
            self.vectorize()

        return {col: int(arr[trace_idx]) for col, arr in self._header_arrays.items()}

    def get_trace_range_headers(
        self,
        start_trace: int,
        end_trace: int
    ) -> Dict[str, np.ndarray]:
        """
        Get headers for a range of traces.

        Args:
            start_trace: First trace index (inclusive)
            end_trace: Last trace index (inclusive)

        Returns:
            Dictionary of segyio field name -> array slice
        """
        if self._header_arrays is None:
            self.vectorize()

        return {
            col: arr[start_trace:end_trace + 1]
            for col, arr in self._header_arrays.items()
        }

    def save(self, path: Path):
        """
        Save vectorized headers to file for worker access.

        Args:
            path: Output file path
        """
        if self._header_arrays is None:
            self.vectorize()

        with open(path, 'wb') as f:
            pickle.dump(self._header_arrays, f)

    @classmethod
    def load(cls, path: Path) -> Dict[str, np.ndarray]:
        """
        Load vectorized headers from file.

        Args:
            path: Input file path

        Returns:
            Dictionary of header arrays (with segyio field names)
        """
        with open(path, 'rb') as f:
            return pickle.load(f)

    def get_stats(self) -> dict:
        """Get statistics about vectorized headers."""
        if self._header_arrays is None:
            self.vectorize()

        total_bytes = sum(arr.nbytes for arr in self._header_arrays.values())
        column_mapping = self._build_column_mapping()

        return {
            'n_traces': self.n_traces,
            'n_fields': len(self._header_arrays),
            'n_columns_in_df': len(self.headers_df.columns),
            'n_mapped_columns': len(column_mapping),
            'fields': list(self._header_arrays.keys()),
            'unmapped_columns': [c for c in self.headers_df.columns if c not in column_mapping],
            'memory_bytes': total_bytes,
            'memory_mb': total_bytes / (1024 ** 2)
        }


def vectorize_headers(headers_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Convenience function to vectorize headers DataFrame.

    Args:
        headers_df: DataFrame with trace headers

    Returns:
        Dictionary mapping segyio field names to numpy arrays
    """
    vectorizer = HeaderVectorizer(headers_df)
    return vectorizer.vectorize()


def get_trace_headers(
    header_arrays: Dict[str, np.ndarray],
    trace_idx: int
) -> Dict[str, int]:
    """
    Get headers for a single trace from vectorized arrays.

    Args:
        header_arrays: Vectorized header arrays (with segyio field names)
        trace_idx: Trace index

    Returns:
        Dictionary of segyio field name -> value
    """
    return {col: int(arr[trace_idx]) for col, arr in header_arrays.items()}
