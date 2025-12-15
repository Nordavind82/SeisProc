"""
Configuration dataclasses for parallel SEG-Y export.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np


@dataclass
class TraceSegment:
    """A segment of traces for parallel export."""
    segment_id: int
    start_trace: int      # First trace index (inclusive)
    end_trace: int        # Last trace index (inclusive)
    n_traces: int         # Number of traces in segment


@dataclass
class ExportHeaderMapping:
    """Header field mapping for export."""
    parquet_column: str      # Column name in headers.parquet
    segy_byte_pos: int       # SEG-Y byte position (1-based)
    format: str              # 'i' for int32, 'h' for int16


@dataclass
class ExportConfig:
    """Configuration for parallel SEG-Y export."""
    processed_zarr_path: str     # Processed traces.zarr path
    headers_parquet_path: str    # Headers parquet path
    output_path: str             # Final output SEG-Y path
    temp_dir: str               # Directory for segment files
    # Optional original SEG-Y (for text/binary header template only)
    original_segy_path: Optional[str] = None
    n_workers: Optional[int] = None  # Auto-detect if None
    chunk_size: int = 10000     # Traces per write operation
    # Export type: 'processed' or 'noise' (input - processed)
    export_type: str = 'processed'
    # Path to input Zarr (required for noise export)
    input_zarr_path: Optional[str] = None
    # Mute configuration (optional)
    mute_velocity: Optional[float] = None    # Mute velocity in m/s
    mute_top: bool = False                   # Apply top mute
    mute_bottom: bool = False                # Apply bottom mute
    mute_taper: int = 20                     # Taper samples
    mute_target: str = 'output'              # 'output', 'input', or 'processed'
    # Custom header mapping (optional - if empty, use all available headers)
    header_mapping: Optional[Dict[str, ExportHeaderMapping]] = None


@dataclass
class ExportTask:
    """Task definition for an export worker process."""
    segment_id: int
    processed_zarr_path: str      # Source trace data
    output_segment_path: str      # Output file for this segment
    header_arrays_path: str       # Path to pickled header arrays
    start_trace: int              # First trace index (inclusive)
    end_trace: int                # Last trace index (inclusive)
    n_samples: int                # Samples per trace
    sample_interval: int          # Sample interval in microseconds
    data_format: int              # SEG-Y data format code
    is_first_segment: bool = False  # First segment includes text/binary headers
    # Optional original SEG-Y (for text/binary header template)
    original_segy_path: Optional[str] = None
    # Export type and noise calculation
    export_type: str = 'processed'         # 'processed' or 'noise'
    input_zarr_path: Optional[str] = None  # Required for noise export
    # Mute configuration (applied on-the-fly)
    mute_velocity: Optional[float] = None  # Mute velocity in m/s
    mute_top: bool = False                 # Apply top mute
    mute_bottom: bool = False              # Apply bottom mute
    mute_taper: int = 20                   # Taper samples
    mute_target: str = 'output'            # 'output', 'input', or 'processed'
    # Custom header mapping (serialized as list of dicts for multiprocessing)
    header_mapping_list: Optional[List[Dict[str, Any]]] = None


@dataclass
class ExportWorkerResult:
    """Result from an export worker process."""
    segment_id: int
    n_traces_exported: int
    output_path: str
    file_size_bytes: int
    elapsed_time: float
    success: bool
    error: Optional[str] = None


@dataclass
class ExportProgress:
    """Progress update from export operation."""
    phase: str                    # 'vectorizing', 'exporting', 'merging', 'finalizing'
    current_traces: int
    total_traces: int
    active_workers: int
    worker_progress: Dict[int, int] = field(default_factory=dict)
    elapsed_time: float = 0.0
    eta_seconds: float = 0.0


@dataclass
class ExportResult:
    """Final result of parallel export operation."""
    success: bool
    output_path: str
    n_traces: int
    n_samples: int
    file_size_bytes: int
    elapsed_time: float
    throughput_traces_per_sec: float
    n_workers_used: int
    error: Optional[str] = None
