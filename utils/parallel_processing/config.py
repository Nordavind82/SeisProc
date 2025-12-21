"""
Configuration dataclasses for parallel processing.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class SortOptions:
    """Options for in-gather trace sorting."""
    enabled: bool = False           # Whether to sort traces within gathers
    sort_key: str = 'offset'        # Header field to sort by
    ascending: bool = True          # Sort direction
    secondary_key: Optional[str] = None  # Optional secondary sort key
    secondary_ascending: bool = True


@dataclass
class ProcessingConfig:
    """Configuration for parallel processing job."""
    input_storage_dir: str      # Directory with traces.zarr, headers.parquet, etc.
    output_storage_dir: str     # Directory for processed output
    processor_config: Dict[str, Any]  # Serialized processor configuration
    n_workers: Optional[int] = None  # Auto-detect if None
    chunk_size: int = 10000     # Traces per processing chunk within gather
    sort_options: Optional[SortOptions] = None  # Optional in-gather sorting
    # Output mode: 'processed' (default), 'noise' (only difference), 'both' (both outputs)
    # 'noise' mode is memory-optimized: outputs only (input - processed) with minimal memory
    output_mode: str = 'processed'
    # Legacy noise output option (deprecated, use output_mode='both' instead)
    output_noise: bool = False  # Also output noise (input - processed)
    # Mute options (applied during processing)
    mute_velocity: float = 0.0  # Mute velocity in m/s (0 = disabled)
    mute_top: bool = False      # Apply top mute
    mute_bottom: bool = False   # Apply bottom mute
    mute_taper: int = 20        # Taper samples for mute transition
    mute_target: str = 'output' # 'output', 'input', or 'processed'


@dataclass
class GatherSegment:
    """A segment of gathers assigned to a worker."""
    segment_id: int
    start_gather: int       # First gather index (inclusive)
    end_gather: int         # Last gather index (inclusive)
    start_trace: int        # Global start trace index
    end_trace: int          # Global end trace index
    n_gathers: int          # Number of gathers in segment
    n_traces: int           # Number of traces in segment


@dataclass
class ProcessingTask:
    """Task definition for a worker process."""
    segment_id: int
    input_zarr_path: str        # Path to input traces.zarr
    output_zarr_path: Optional[str]  # Path to shared output traces.zarr (None for noise-only mode)
    headers_parquet_path: str   # Path to headers.parquet
    ensemble_index_path: str    # Path to ensemble_index.parquet
    processor_config: Dict[str, Any]  # Serialized processor config
    start_gather: int           # First gather index (inclusive)
    end_gather: int             # Last gather index (inclusive)
    n_samples: int              # Samples per trace
    sample_rate: float          # For SeismicData construction
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    sort_options: Optional[SortOptions] = None  # Optional in-gather sorting
    sort_mapping_path: Optional[str] = None  # Path to write sort mapping
    # Output mode: 'processed', 'noise', or 'both'
    output_mode: str = 'processed'
    # Legacy noise output options (use output_mode='both' for new code)
    output_noise: bool = False  # Also output noise (input - processed)
    noise_zarr_path: Optional[str] = None  # Path to noise output zarr
    # Mute options
    mute_velocity: float = 0.0  # Mute velocity in m/s (0 = disabled)
    mute_top: bool = False      # Apply top mute
    mute_bottom: bool = False   # Apply bottom mute
    mute_taper: int = 20        # Taper samples for mute transition
    mute_target: str = 'output' # 'output', 'input', or 'processed'


@dataclass
class ProcessingWorkerResult:
    """Result from a worker process."""
    segment_id: int
    n_gathers_processed: int
    n_traces_processed: int
    elapsed_time: float
    success: bool
    error: Optional[str] = None
    sort_mapping_path: Optional[str] = None  # Path to sort mapping file if sorting was applied


@dataclass
class ProcessingProgress:
    """Progress information for callbacks."""
    phase: str                  # 'initializing', 'processing', 'finalizing'
    current_traces: int
    total_traces: int
    current_gathers: int
    total_gathers: int
    active_workers: int
    worker_progress: Dict[int, int] = field(default_factory=dict)
    elapsed_time: float = 0.0
    eta_seconds: float = 0.0
    traces_per_sec: float = 0.0
    compute_kernel: str = ""


@dataclass
class ProcessingResult:
    """Final result of processing operation."""
    success: bool
    output_dir: str
    output_zarr_path: str
    n_gathers: int
    n_traces: int
    n_samples: int
    elapsed_time: float
    throughput_traces_per_sec: float
    n_workers_used: int
    error: Optional[str] = None
    noise_zarr_path: Optional[str] = None  # Path to noise output if generated
