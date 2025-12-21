"""
Coordinator for parallel multiprocess gather processing.

DEPRECATED: This module uses ProcessPoolExecutor which has issues with
Metal GPU initialization in child processes. Use the Ray-based
RayProcessingCoordinator from utils.ray_orchestration instead:

    from utils.ray_orchestration import RayProcessingCoordinator
    coordinator = RayProcessingCoordinator(config)
    result = coordinator.run()

Orchestrates the full processing pipeline:
1. Validate inputs and load metadata
2. Pre-create shared output Zarr array
3. Partition gathers across workers
4. Launch and monitor worker processes
5. If sorting enabled, create sorted headers from worker mappings
6. Handle progress updates and cancellation
"""

import os
import gc
import json
import time
import shutil
import pickle
import numpy as np
import zarr
import pandas as pd
import psutil
import logging
import traceback
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from datetime import datetime


# =============================================================================
# DEBUG LOGGING - Writes immediately to file with flush for crash diagnosis
# =============================================================================
_DEBUG_LOG_FILE = None
_DEBUG_ENABLED = True  # Set to False to disable debug logging


def _init_debug_log(output_dir: Optional[Path] = None):
    """Initialize debug log file."""
    global _DEBUG_LOG_FILE
    if not _DEBUG_ENABLED:
        return
    try:
        if output_dir:
            log_path = output_dir / 'parallel_processing_debug.log'
        else:
            log_path = Path('/tmp/parallel_processing_debug.log')
        _DEBUG_LOG_FILE = open(log_path, 'w')
        _debug_log(f"Debug log initialized at {datetime.now().isoformat()}")
        _debug_log(f"Log file: {log_path}")
    except Exception as e:
        print(f"Warning: Could not initialize debug log: {e}")


def _debug_log(msg: str, include_memory: bool = False):
    """Write debug message immediately to file with flush."""
    global _DEBUG_LOG_FILE
    if not _DEBUG_ENABLED or _DEBUG_LOG_FILE is None:
        return
    try:
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        mem_info = ""
        if include_memory:
            mem = psutil.virtual_memory()
            mem_info = f" [MEM: {mem.used/(1024**3):.2f}GB used / {mem.available/(1024**3):.2f}GB avail]"
        line = f"[{timestamp}] COORD: {msg}{mem_info}\n"
        _DEBUG_LOG_FILE.write(line)
        _DEBUG_LOG_FILE.flush()  # Immediate flush for crash diagnosis
        os.fsync(_DEBUG_LOG_FILE.fileno())  # Force OS to write to disk
    except Exception:
        pass


def _close_debug_log():
    """Close debug log file."""
    global _DEBUG_LOG_FILE
    if _DEBUG_LOG_FILE is not None:
        try:
            _debug_log("Debug log closing")
            _DEBUG_LOG_FILE.close()
        except Exception:
            pass
        _DEBUG_LOG_FILE = None

from .config import (
    ProcessingConfig,
    ProcessingTask,
    ProcessingWorkerResult,
    ProcessingProgress,
    ProcessingResult,
    GatherSegment,
    SortOptions
)
from .partitioner import GatherPartitioner
from .worker import process_gather_range, read_streaming_sort_file
from .shared_data import (
    set_shared_headers,
    set_shared_ensemble_index,
    clear_shared_data,
    get_shared_data_summary
)
from utils.parquet_io import read_parquet, read_parquet_schema, find_columns_case_insensitive

# Import settings for memory configuration
try:
    from models.app_settings import get_settings
    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False

logger = logging.getLogger(__name__)


def get_optimal_workers() -> int:
    """Get optimal number of worker processes."""
    cpu_count = os.cpu_count() or 4
    # Leave 2 cores free for system/UI, minimum 2 workers
    return max(2, cpu_count - 2)


def check_parallel_processing_memory_budget(
    n_workers: int,
    max_gather_traces: int,
    n_samples: int,
    memory_copies: int = 4,
    safety_factor: float = 0.7,
    output_noise: bool = False,
    sorting_enabled: bool = False,
    output_mode: str = 'processed'
) -> Tuple[bool, float, float, float, str, str]:
    """
    Pre-flight memory check for parallel gather processing.

    Calculates: n_workers × max_gather_size × memory_copies × 4 bytes (float32)
    PLUS worker startup overhead (imports, ensemble index, etc.)

    Memory usage by mode:
    - 'processed': ~2 copies (input + processed)
    - 'noise': ~2 copies (input reused for result, + processed during computation)
    - 'both': ~3-4 copies (input + processed + noise)

    Args:
        n_workers: Number of parallel worker processes
        max_gather_traces: Maximum traces in any single gather
        n_samples: Number of samples per trace
        memory_copies: Base estimated copies per gather (2-8)
        safety_factor: Fraction of available memory considered safe (0.0-1.0)
        output_noise: Whether noise output is enabled (legacy, adds 1 copy)
        sorting_enabled: Whether sorting is enabled (adds 1 copy)
        output_mode: 'processed', 'noise', or 'both'

    Returns:
        Tuple of (is_safe, available_mb, required_mb, ratio, risk_level, message)
        risk_level: 'low', 'medium', 'high', 'critical'
    """
    available_bytes = psutil.virtual_memory().available
    available_mb = available_bytes / (1024**2)

    # Per-worker memory overhead (MB) - MEASURED VALUES from actual runs
    # Components that add up per worker:
    #   - Fork base overhead: ~200 MB (Python pages that get modified, triggering COW)
    #   - Ensemble index: ~80 MB (loaded AFTER fork, so NOT shared via COW)
    #   - Zarr metadata/cache: ~250 MB (input + output arrays, chunk cache)
    #   - Working buffers: ~150 MB (numpy temp arrays during processing)
    #   - DWT/SWT peak: ~100 MB additional during wavelet transform
    # Total measured: ~780 MB per worker (without headers)
    #
    # Headers for mute/sorting: ~170 MB per worker IF NOT pre-shared
    # With fork COW pre-sharing (Linux), headers add ~0 MB per worker
    import sys
    if sys.platform == 'linux':
        WORKER_STARTUP_OVERHEAD_MB = 800  # Fork: measured ~780 MB per worker
    else:
        WORKER_STARTUP_OVERHEAD_MB = 2500  # Spawn: full import overhead

    # Determine effective output mode
    effective_mode = output_mode
    if output_mode == 'processed' and output_noise:
        effective_mode = 'both'

    # Calculate copies based on output mode
    # NOTE: Peak memory occurs DURING processor.process(), not after!
    # DWT/SWT processors internally create during processing:
    #   1. Input gather (from zarr)
    #   2. Padded array (for SWT power-of-2 requirement)
    #   3. Denoised/output array
    #   4. Coefficient arrays during SWT (temporary but significant)
    # Total peak: ~4 copies for DWT/SWT processors
    #
    # After processing returns, worker does in-place operations to minimize memory
    # But we must budget for the processor's peak, not the post-processing state
    if effective_mode == 'noise':
        # Peak during processor: input + padded + denoised + coefficients
        # Post-processing is optimized with in-place subtraction
        effective_copies = 4  # Measured peak during DWT processing
    elif effective_mode == 'both':
        # Peak: input + processor_internals + separate noise array
        effective_copies = 5
    else:
        # Processed only: input + processor_internals
        effective_copies = 4

    if sorting_enabled:
        effective_copies += 1

    # Calculate memory per gather: n_samples × max_traces × 4 bytes × copies
    bytes_per_gather = n_samples * max_gather_traces * 4 * effective_copies

    # Total peak memory: all workers loading max-size gathers simultaneously
    peak_bytes = n_workers * bytes_per_gather
    gather_mb = peak_bytes / (1024**2)

    # Add worker startup overhead (imports, ensemble index, zarr handles)
    worker_overhead_mb = n_workers * WORKER_STARTUP_OVERHEAD_MB
    required_mb = gather_mb + worker_overhead_mb

    # Calculate safe number of workers based on available memory
    # available_mb = n_workers * (startup_overhead + gather_memory_per_worker)
    gather_mb_per_worker = gather_mb / n_workers if n_workers > 0 else 0
    per_worker_total_mb = WORKER_STARTUP_OVERHEAD_MB + gather_mb_per_worker
    safe_workers = max(1, int(available_mb * safety_factor / per_worker_total_mb)) if per_worker_total_mb > 0 else 1

    # Calculate ratio
    ratio = required_mb / available_mb if available_mb > 0 else float('inf')
    safe_threshold = safety_factor

    # Determine risk level
    if ratio < safe_threshold * 0.5:
        risk_level = 'low'
        is_safe = True
    elif ratio < safe_threshold:
        risk_level = 'medium'
        is_safe = True
    elif ratio < 0.9:
        risk_level = 'high'
        is_safe = False
    else:
        risk_level = 'critical'
        is_safe = False

    # Build message with mode info
    mode_label = effective_mode.upper()
    if effective_mode == 'noise':
        mode_label += " (memory-optimized)"

    if risk_level == 'low':
        message = (
            f"Memory OK [{mode_label}]: {n_workers} workers × ~{per_worker_total_mb:.0f} MB/worker = "
            f"{required_mb:.0f} MB ({ratio*100:.0f}% of {available_mb:.0f} MB available)"
        )
    elif risk_level == 'medium':
        message = (
            f"Memory MODERATE [{mode_label}]: {required_mb:.0f} MB estimated vs {available_mb:.0f} MB available "
            f"({ratio*100:.0f}%). Processing should work but monitor system memory."
        )
    elif risk_level == 'high':
        message = (
            f"Memory WARNING [{mode_label}]: {required_mb:.0f} MB estimated vs {available_mb:.0f} MB available "
            f"({ratio*100:.0f}%). Reduce workers to {safe_workers} or less."
        )
    else:  # critical
        message = (
            f"Memory CRITICAL [{mode_label}]: {required_mb:.0f} MB estimated vs {available_mb:.0f} MB available "
            f"({ratio*100:.0f}%). WILL CRASH! Reduce workers to {safe_workers} or less."
        )

    # Return tuple with safe_workers as additional info
    return is_safe, available_mb, required_mb, ratio, risk_level, message, safe_workers


def check_sorting_memory_budget(
    n_traces: int,
    n_header_columns: int = 50,
    safety_factor: float = 0.7
) -> Tuple[bool, float, float, str]:
    """
    Pre-flight memory check before sorting operation.

    Args:
        n_traces: Number of traces to process
        n_header_columns: Estimated number of header columns
        safety_factor: Fraction of available memory considered safe (default 70%)

    Returns:
        Tuple of (is_safe, available_mb, required_mb, message)
    """
    available_bytes = psutil.virtual_memory().available
    available_mb = available_bytes / (1024**2)

    # Estimate memory requirements for sorting
    # 1. Global mapping array: int64 per trace
    mapping_mb = (n_traces * 8) / (1024**2)

    # 2. Headers DataFrame (estimate ~8 bytes per value average)
    headers_mb = (n_header_columns * n_traces * 8) / (1024**2)

    # 3. Sorted headers copy during reorder
    sorted_copy_mb = headers_mb

    # Total peak estimate
    required_mb = mapping_mb + headers_mb + sorted_copy_mb

    safe_available = available_mb * safety_factor
    is_safe = required_mb < safe_available

    if is_safe:
        message = (
            f"Memory OK: ~{required_mb:.0f} MB required for sorting, "
            f"{available_mb:.0f} MB available"
        )
    else:
        message = (
            f"MEMORY WARNING: Sorting requires ~{required_mb:.0f} MB, "
            f"only {available_mb:.0f} MB available ({safety_factor*100:.0f}% threshold). "
            f"Consider disabling sorting or processing smaller batches."
        )

    return is_safe, available_mb, required_mb, message


class ParallelProcessingCoordinator:
    """
    Orchestrates parallel multiprocess gather processing.

    .. deprecated::
        This class uses ProcessPoolExecutor which has issues with Metal GPU
        initialization in forked processes. Use RayProcessingCoordinator from
        utils.ray_orchestration instead, which properly initializes GPU
        contexts in each Ray actor.

    Workers write directly to a pre-created shared Zarr array,
    eliminating the need for output merging.

    When sorting is enabled, traces are sorted within each gather
    and a sorted headers.parquet is created for export.

    Usage (deprecated):
        config = ProcessingConfig(...)
        coordinator = ParallelProcessingCoordinator(config)
        result = coordinator.run(progress_callback=update_ui)

    New Usage (preferred):
        from utils.ray_orchestration import RayProcessingCoordinator
        coordinator = RayProcessingCoordinator(config)
        result = coordinator.run()
    """

    def __init__(self, config: ProcessingConfig):
        """
        Initialize coordinator.

        Args:
            config: Processing configuration
        """
        import warnings
        warnings.warn(
            "ParallelProcessingCoordinator is deprecated. Use "
            "RayProcessingCoordinator from utils.ray_orchestration instead, "
            "which properly supports Metal GPU initialization per worker.",
            DeprecationWarning,
            stacklevel=2
        )
        self.config = config
        self.n_workers = config.n_workers or get_optimal_workers()
        self._cancel_requested = False
        self._executor = None  # Store executor reference for cancellation
        self._futures = {}  # Store futures for cancellation

    def run(
        self,
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None
    ) -> ProcessingResult:
        """
        Run the full parallel processing pipeline.

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            ProcessingResult with outcome
        """
        # EARLY CRASH DIAGNOSTIC - write to /tmp before any other operations
        def _early_crash_log(msg: str):
            """Write to shared crash log for diagnosis."""
            try:
                import os
                from datetime import datetime
                with open('/tmp/seisproc_parallel_crash.log', 'a') as f:
                    ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                    mem = psutil.virtual_memory()
                    f.write(f"[{ts}] [MEM: {mem.used/(1024**3):.2f}GB/{mem.available/(1024**3):.2f}GB] COORD: {msg}\n")
                    f.flush()
                    os.fsync(f.fileno())
            except Exception:
                pass

        _early_crash_log(">>> run() method ENTERED")

        start_time = time.time()
        _early_crash_log(">>> start_time set")

        input_dir = Path(self.config.input_storage_dir)
        _early_crash_log(f">>> input_dir={input_dir}")

        output_dir = Path(self.config.output_storage_dir)
        _early_crash_log(f">>> output_dir={output_dir}")
        sorting_enabled = (
            self.config.sort_options is not None and
            self.config.sort_options.enabled
        )
        _early_crash_log(f">>> sorting_enabled={sorting_enabled}")

        # Initialize debug logging FIRST
        _early_crash_log(">>> Creating output directory...")
        output_dir.mkdir(parents=True, exist_ok=True)
        _early_crash_log(">>> Output directory created, initializing debug log...")
        _init_debug_log(output_dir)
        _early_crash_log(">>> Debug log initialized")
        _debug_log("=" * 60)
        _debug_log("PARALLEL PROCESSING COORDINATOR STARTING", include_memory=True)
        _debug_log(f"Input dir: {input_dir}")
        _debug_log(f"Output dir: {output_dir}")
        _debug_log(f"Output mode: {getattr(self.config, 'output_mode', 'processed')}")
        _debug_log(f"N workers requested: {self.n_workers}")
        _debug_log(f"Sorting enabled: {sorting_enabled}")

        try:
            # Phase 1: Validate and load metadata
            _early_crash_log(">>> Phase 1: Validating and loading metadata...")
            _debug_log("Phase 1: Validating and loading metadata...", include_memory=True)
            if progress_callback:
                _early_crash_log(">>> Calling progress_callback for initializing...")
                progress_callback(ProcessingProgress(
                    phase='initializing',
                    current_traces=0,
                    total_traces=0,
                    current_gathers=0,
                    total_gathers=0,
                    active_workers=0
                ))
                _early_crash_log(">>> progress_callback returned")

            _early_crash_log(">>> Calling _load_and_validate...")
            metadata, ensemble_df = self._load_and_validate(input_dir)
            _early_crash_log(f">>> _load_and_validate returned, {len(ensemble_df)} gathers")
            _debug_log(f"Metadata loaded: n_traces={metadata.get('n_traces')}, n_samples={metadata.get('n_samples')}", include_memory=True)
            _debug_log(f"Ensemble index loaded: {len(ensemble_df)} gathers")
            n_traces = metadata['n_traces']
            n_samples = metadata['n_samples']
            n_gathers = len(ensemble_df)
            sample_rate = metadata['sample_rate']

            # Get max gather size for memory estimation
            max_gather_traces = int(ensemble_df['n_traces'].max())
            avg_gather_traces = int(ensemble_df['n_traces'].mean())
            _debug_log(f"Gather stats: max={max_gather_traces} traces, avg={avg_gather_traces} traces")

            # Determine output mode early for memory check
            output_mode_check = getattr(self.config, 'output_mode', 'processed')
            if output_mode_check == 'processed' and self.config.output_noise:
                output_mode_check = 'both'
            _debug_log(f"Effective output mode: {output_mode_check}")

            # MEMORY GUARD: Pre-flight check for parallel processing memory
            _debug_log("Running memory budget check...", include_memory=True)
            memory_check_result = self._check_memory_budget(
                n_workers=self.n_workers,
                max_gather_traces=max_gather_traces,
                n_samples=n_samples,
                sorting_enabled=sorting_enabled,
                output_noise=self.config.output_noise,
                output_mode=output_mode_check
            )

            if memory_check_result is not None:
                _debug_log(f"Memory check result: {memory_check_result}")
                is_safe, available_mb, required_mb, ratio, risk_level, mem_message, safe_workers = memory_check_result
                logger.info(mem_message)
                print(f"  {mem_message}")

                # Check if we should block execution
                if not is_safe and risk_level == 'critical':
                    if SETTINGS_AVAILABLE:
                        settings = get_settings()
                        if settings.get_parallel_block_on_high_risk():
                            raise MemoryError(
                                f"Parallel processing blocked due to critical memory risk.\n"
                                f"Estimated: {required_mb:.0f} MB, Available: {available_mb:.0f} MB\n"
                                f"Max gather: {max_gather_traces:,} traces\n"
                                f"Reduce workers from {self.n_workers} to "
                                f"{max(1, int(self.n_workers * available_mb * 0.7 / required_mb))} or less.\n"
                                f"Or disable 'Block on high risk' in Settings > Performance."
                            )

            # Determine output mode
            output_mode = getattr(self.config, 'output_mode', 'processed')
            if output_mode == 'processed' and self.config.output_noise:
                output_mode = 'both'  # Legacy compatibility

            output_processed = output_mode in ('processed', 'both')
            output_noise_flag = output_mode in ('noise', 'both')

            sort_info = ""
            if sorting_enabled:
                sort_info = f" (sorting by {self.config.sort_options.sort_key})"

                # MEMORY GUARD: Check if we have enough memory for sorting
                is_safe, available_mb, required_mb, mem_message = check_sorting_memory_budget(
                    n_traces=n_traces,
                    safety_factor=0.7
                )
                logger.info(mem_message)

                if not is_safe:
                    # Log warning but allow to proceed (user can abort if needed)
                    logger.warning(
                        f"Low memory warning for sorting operation. "
                        f"Required: ~{required_mb:.0f}MB, Available: {available_mb:.0f}MB. "
                        f"Processing will continue but may fail."
                    )
                    print(f"  WARNING: {mem_message}")

            mode_info = ""
            if output_mode == 'noise':
                mode_info = " [NOISE-ONLY mode - memory optimized]"
            elif output_mode == 'both':
                mode_info = " [processed + noise output]"

            print(f"  Processing {n_gathers:,} gathers, {n_traces:,} traces{sort_info}{mode_info}")
            _debug_log(f"Processing {n_gathers:,} gathers, {n_traces:,} traces{sort_info}{mode_info}")

            # Phase 2: Partition gathers across workers
            _early_crash_log(">>> Phase 2: Partitioning gathers across workers...")
            _debug_log("Phase 2: Partitioning gathers across workers...", include_memory=True)
            segments = self._partition_gathers(ensemble_df)
            _early_crash_log(f">>> Partitioned into {len(segments)} segments")
            print(f"  Partitioned into {len(segments)} segments")
            _debug_log(f"Partitioned into {len(segments)} segments")
            for seg in segments:
                _debug_log(f"  Segment {seg.segment_id}: gathers {seg.start_gather}-{seg.end_gather}, {seg.n_traces:,} traces")

            # Phase 3: Create output directory and shared Zarr
            _early_crash_log(">>> Phase 3: Creating output Zarr arrays...")
            _debug_log("Phase 3: Creating output Zarr arrays...", include_memory=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Create output zarr only if outputting processed data
            output_zarr_path = None
            if output_processed:
                _early_crash_log(f">>> Creating OUTPUT zarr ({n_samples} x {n_traces:,})...")
                _debug_log(f"Creating OUTPUT zarr ({n_samples} x {n_traces:,})...", include_memory=True)
                output_zarr_path = self._create_output_zarr(output_dir, n_samples, n_traces)
                _early_crash_log(f">>> Output zarr created at {output_zarr_path}")
                print(f"  Created output Zarr: {n_samples} x {n_traces:,}")
                _debug_log(f"Output zarr created: {output_zarr_path}", include_memory=True)
            else:
                _early_crash_log(">>> Skipping output zarr (noise-only mode)")
                _debug_log("Skipping output zarr (noise-only mode)")

            # Create noise zarr if noise output is enabled
            noise_zarr_path = None
            if output_noise_flag:
                _early_crash_log(f">>> Creating NOISE zarr ({n_samples} x {n_traces:,})...")
                _debug_log(f"Creating NOISE zarr ({n_samples} x {n_traces:,})...", include_memory=True)
                noise_zarr_path = self._create_noise_zarr(output_dir, n_samples, n_traces)
                _early_crash_log(f">>> Noise zarr created at {noise_zarr_path}")
                print(f"  Created noise Zarr: {n_samples} x {n_traces:,}")
                _debug_log(f"Noise zarr created: {noise_zarr_path}", include_memory=True)

            # Create temp directory for sort mappings if sorting enabled
            temp_dir = None
            if sorting_enabled:
                temp_dir = output_dir / 'temp_sort'
                temp_dir.mkdir(parents=True, exist_ok=True)
                _debug_log(f"Sort temp directory: {temp_dir}")

            # Phase 4: Run parallel workers
            _early_crash_log(">>> Phase 4: Launching parallel workers...")
            _debug_log("Phase 4: Launching parallel workers...", include_memory=True)
            _early_crash_log(f">>> About to start {len(segments)} workers with ProcessPoolExecutor")
            _debug_log(f"About to start {len(segments)} workers with ProcessPoolExecutor")
            if progress_callback:
                progress_callback(ProcessingProgress(
                    phase='processing',
                    current_traces=0,
                    total_traces=n_traces,
                    current_gathers=0,
                    total_gathers=n_gathers,
                    active_workers=len(segments)
                ))

            worker_results = self._run_workers(
                segments=segments,
                input_dir=input_dir,
                output_zarr_path=output_zarr_path,
                noise_zarr_path=noise_zarr_path,
                n_samples=n_samples,
                sample_rate=sample_rate,
                metadata=metadata,
                n_traces=n_traces,
                n_gathers=n_gathers,
                temp_dir=temp_dir,
                progress_callback=progress_callback
            )

            # Check for failures
            failed = [r for r in worker_results if not r.success]
            if failed:
                errors = "; ".join([f"Segment {r.segment_id}: {r.error}" for r in failed])
                raise RuntimeError(f"Worker failures: {errors}")

            # Phase 5: Create sorted headers if sorting was enabled
            if progress_callback:
                progress_callback(ProcessingProgress(
                    phase='finalizing',
                    current_traces=n_traces,
                    total_traces=n_traces,
                    current_gathers=n_gathers,
                    total_gathers=n_gathers,
                    active_workers=0
                ))

            if sorting_enabled:
                print("  Creating sorted headers...")
                self._create_sorted_headers(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    worker_results=worker_results,
                    ensemble_df=ensemble_df,
                    n_traces=n_traces
                )
                # Cleanup temp directory
                if temp_dir and temp_dir.exists():
                    shutil.rmtree(temp_dir)
            else:
                # Copy metadata files without sorting
                self._copy_metadata_files(input_dir, output_dir, metadata)

            # Update metadata with processing info
            self._save_processing_metadata(output_dir, metadata, len(segments), sorting_enabled)

            elapsed = time.time() - start_time
            throughput = n_traces / elapsed if elapsed > 0 else 0

            noise_info = ""
            if output_mode == 'noise':
                noise_info = " [noise-only output]"
            elif noise_zarr_path:
                noise_info = " + noise output"
            print(f"  Processing complete: {n_traces:,} traces in {elapsed:.1f}s "
                  f"({throughput:,.0f} traces/sec){noise_info}")

            # For noise-only mode, create a symlink traces.zarr -> noise.zarr
            # This allows standard loading code (LazySeismicData.from_storage_dir) to work
            if output_mode == 'noise' and noise_zarr_path:
                traces_symlink = output_dir / 'traces.zarr'
                noise_zarr_actual = output_dir / 'noise.zarr'
                if not traces_symlink.exists() and noise_zarr_actual.exists():
                    try:
                        traces_symlink.symlink_to('noise.zarr')
                        _debug_log(f"Created symlink traces.zarr -> noise.zarr for loading compatibility")
                    except Exception as e:
                        _debug_log(f"Warning: Could not create traces.zarr symlink: {e}")

            # For noise-only mode, the main output is the noise zarr
            # For backwards compatibility, output_zarr_path is the processed output (or noise if no processed)
            main_output_path = output_zarr_path if output_zarr_path else noise_zarr_path

            # Validate all output files before declaring success
            self._validate_output_files(output_dir, output_mode)

            # Log successful completion
            _debug_log("=" * 60)
            _debug_log("PARALLEL PROCESSING COMPLETED SUCCESSFULLY", include_memory=True)
            _debug_log(f"Total gathers: {n_gathers:,}")
            _debug_log(f"Total traces: {n_traces:,}")
            _debug_log(f"Elapsed time: {elapsed:.1f}s")
            _debug_log(f"Throughput: {throughput:,.0f} traces/sec")
            _debug_log(f"Output path: {main_output_path}")
            _debug_log("=" * 60)
            _close_debug_log()

            return ProcessingResult(
                success=True,
                output_dir=str(output_dir),
                output_zarr_path=main_output_path,
                n_gathers=n_gathers,
                n_traces=n_traces,
                n_samples=n_samples,
                elapsed_time=elapsed,
                throughput_traces_per_sec=throughput,
                n_workers_used=len(segments),
                noise_zarr_path=noise_zarr_path
            )

        except Exception as e:
            elapsed = time.time() - start_time
            error_trace = traceback.format_exc()

            # Log the error to debug file
            _debug_log("=" * 60)
            _debug_log("EXCEPTION CAUGHT IN COORDINATOR", include_memory=True)
            _debug_log(f"Exception type: {type(e).__name__}")
            _debug_log(f"Exception message: {str(e)}")
            _debug_log(f"Full traceback:\n{error_trace}")
            _debug_log("=" * 60)
            _close_debug_log()

            # Cleanup on failure
            if output_dir.exists():
                try:
                    shutil.rmtree(output_dir)
                except Exception:
                    pass

            return ProcessingResult(
                success=False,
                output_dir=str(output_dir),
                output_zarr_path="",
                n_gathers=0,
                n_traces=0,
                n_samples=0,
                elapsed_time=elapsed,
                throughput_traces_per_sec=0,
                n_workers_used=0,
                error=f"{str(e)}\n{error_trace}"
            )

    def _load_and_validate(self, input_dir: Path) -> tuple:
        """Load and validate input data."""
        # Check required files
        zarr_path = input_dir / 'traces.zarr'
        metadata_path = input_dir / 'metadata.json'
        ensemble_path = input_dir / 'ensemble_index.parquet'

        if not zarr_path.exists():
            raise FileNotFoundError(f"Input Zarr not found: {zarr_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        if not ensemble_path.exists():
            raise FileNotFoundError(f"Ensemble index not found: {ensemble_path}")

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Load ensemble index
        ensemble_df = pd.read_parquet(ensemble_path)

        # Validate
        if 'n_traces' not in metadata or 'n_samples' not in metadata:
            raise ValueError("Metadata missing required fields (n_traces, n_samples)")

        required_cols = ['start_trace', 'end_trace', 'n_traces']
        missing = [c for c in required_cols if c not in ensemble_df.columns]
        if missing:
            raise ValueError(f"Ensemble index missing columns: {missing}")

        return metadata, ensemble_df

    def _partition_gathers(self, ensemble_df: pd.DataFrame) -> List[GatherSegment]:
        """Partition gathers across workers."""
        partitioner = GatherPartitioner(ensemble_df, self.n_workers)
        segments = partitioner.partition()

        # Log partition stats
        stats = partitioner.get_partition_stats(segments)
        print(f"  Partition stats: {stats}")

        return segments

    def _create_output_zarr(self, output_dir: Path, n_samples: int, n_traces: int) -> str:
        """Create pre-allocated output Zarr array."""
        output_path = output_dir / 'traces.zarr'

        zarr.open(
            str(output_path),
            mode='w',
            shape=(n_samples, n_traces),
            chunks=(n_samples, 1000),
            dtype=np.float32,
            compressor=None,  # No compression for speed
            zarr_format=2
        )

        return str(output_path)

    def _create_noise_zarr(self, output_dir: Path, n_samples: int, n_traces: int) -> str:
        """Create pre-allocated noise output Zarr array."""
        noise_path = output_dir / 'noise.zarr'

        zarr.open(
            str(noise_path),
            mode='w',
            shape=(n_samples, n_traces),
            chunks=(n_samples, 1000),
            dtype=np.float32,
            compressor=None,  # No compression for speed
            zarr_format=2
        )

        return str(noise_path)

    def _copy_metadata_files(self, input_dir: Path, output_dir: Path, metadata: dict):
        """Copy metadata and index files to output.

        Raises
        ------
        FileNotFoundError
            If required metadata files are missing from input
        RuntimeError
            If file copy operations fail
        """
        # Required files - raise error if missing
        required_files = [
            ('headers.parquet', 'Trace headers'),
            ('ensemble_index.parquet', 'Ensemble/gather index'),
        ]

        for filename, description in required_files:
            src = input_dir / filename
            dst = output_dir / filename

            if not src.exists():
                raise FileNotFoundError(
                    f"Required metadata file missing: {filename} ({description}). "
                    f"Input dataset at {input_dir} may be corrupted or incomplete. "
                    f"Try re-importing the original SEG-Y file."
                )

            try:
                shutil.copy2(src, dst)
                # Verify copy succeeded
                if not dst.exists():
                    raise RuntimeError(f"File copy failed: {filename} not created at {dst}")
                _debug_log(f"Copied {filename} to output")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to copy {filename}: {e}. "
                    f"Check disk space and permissions."
                ) from e

        # Optional files - copy if exists, log if missing
        optional_files = ['trace_index.parquet']

        for filename in optional_files:
            src = input_dir / filename
            dst = output_dir / filename

            if src.exists():
                try:
                    shutil.copy2(src, dst)
                    _debug_log(f"Copied optional file {filename}")
                except Exception as e:
                    logger.warning(f"Failed to copy optional file {filename}: {e}")
            else:
                _debug_log(f"Optional file {filename} not present in input, skipping")

    def _run_workers(
        self,
        segments: List[GatherSegment],
        input_dir: Path,
        output_zarr_path: Optional[str],
        noise_zarr_path: Optional[str],
        n_samples: int,
        sample_rate: float,
        metadata: dict,
        n_traces: int,
        n_gathers: int,
        temp_dir: Optional[Path],
        progress_callback: Optional[Callable]
    ) -> List[ProcessingWorkerResult]:
        """Run worker processes in parallel."""
        # Early crash logging function
        def _early_log(msg: str):
            try:
                import os
                from datetime import datetime
                with open('/tmp/seisproc_parallel_crash.log', 'a') as f:
                    ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                    mem = psutil.virtual_memory()
                    f.write(f"[{ts}] [MEM: {mem.used/(1024**3):.2f}GB/{mem.available/(1024**3):.2f}GB] WORKERS: {msg}\n")
                    f.flush()
                    os.fsync(f.fileno())
            except Exception:
                pass

        _early_log(">>> _run_workers() ENTERED")
        _debug_log("_run_workers() called", include_memory=True)
        sorting_enabled = (
            self.config.sort_options is not None and
            self.config.sort_options.enabled
        )

        # Determine output mode
        output_mode = getattr(self.config, 'output_mode', 'processed')
        if output_mode == 'processed' and self.config.output_noise:
            output_mode = 'both'
        _debug_log(f"Output mode for workers: {output_mode}")
        _debug_log(f"output_zarr_path: {output_zarr_path}")
        _debug_log(f"noise_zarr_path: {noise_zarr_path}")

        # Create worker tasks
        _debug_log("Creating worker tasks...", include_memory=True)
        tasks = []
        for segment in segments:
            # Set up sort mapping path if sorting enabled
            sort_mapping_path = None
            if sorting_enabled and temp_dir:
                sort_mapping_path = str(temp_dir / f'sort_mapping_{segment.segment_id}.pkl')

            task = ProcessingTask(
                segment_id=segment.segment_id,
                input_zarr_path=str(input_dir / 'traces.zarr'),
                output_zarr_path=output_zarr_path,
                headers_parquet_path=str(input_dir / 'headers.parquet'),
                ensemble_index_path=str(input_dir / 'ensemble_index.parquet'),
                processor_config=self.config.processor_config,
                start_gather=segment.start_gather,
                end_gather=segment.end_gather,
                n_samples=n_samples,
                sample_rate=sample_rate,
                metadata=metadata,
                sort_options=self.config.sort_options if sorting_enabled else None,
                sort_mapping_path=sort_mapping_path,
                # Output mode (new field)
                output_mode=output_mode,
                # Legacy noise output options (for backwards compatibility)
                output_noise=self.config.output_noise,
                noise_zarr_path=noise_zarr_path,
                # Mute options
                mute_velocity=self.config.mute_velocity,
                mute_top=self.config.mute_top,
                mute_bottom=self.config.mute_bottom,
                mute_taper=self.config.mute_taper,
                mute_target=self.config.mute_target
            )
            tasks.append(task)

        # Use multiprocessing Manager for progress queue
        _early_log(">>> Creating multiprocessing Manager...")
        _debug_log("Creating multiprocessing Manager and progress queue...", include_memory=True)
        manager = mp.Manager()
        _early_log(">>> Manager created, creating queue...")
        progress_queue = manager.Queue()
        _early_log(">>> Queue created successfully")
        _debug_log("Manager created successfully", include_memory=True)

        # Track progress per worker
        worker_progress_traces = {s.segment_id: 0 for s in segments}
        worker_progress_gathers = {s.segment_id: 0 for s in segments}
        results = []
        processed_futures = set()

        print(f"  Launching {len(tasks)} worker processes...")
        _early_log(f">>> About to create ProcessPoolExecutor with max_workers={self.n_workers}")
        _debug_log(f"About to create ProcessPoolExecutor with max_workers={self.n_workers}", include_memory=True)

        # Use FORK context on Linux for copy-on-write memory sharing
        # Fork does NOT copy memory - it shares read-only pages (numpy, scipy, pandas)
        # Only written pages get duplicated, so workers share ~1.5GB of imports
        # This allows 14+ workers on 32GB systems vs only 6-8 with spawn
        #
        # Spawn creates fresh processes that must re-import everything (~2GB each)
        # Use spawn only on macOS (fork unsafe with Objective-C) or Windows (no fork)
        import sys
        if sys.platform == 'linux':
            mp_ctx = mp.get_context('fork')
            _early_log(">>> Using 'fork' context for copy-on-write memory sharing (Linux)")

            # CRITICAL: Reset GPU DeviceManager singleton before fork!
            # The UI (control_panel.py) initializes CUDA when checking GPU availability.
            # If we fork with an initialized DeviceManager, workers inherit a broken
            # CUDA context. Resetting it ensures workers create fresh CUDA contexts.
            try:
                from processors.gpu.device_manager import reset_device_manager
                reset_device_manager()
                _early_log(">>> Reset GPU DeviceManager singleton before fork (workers will create fresh CUDA context)")
            except ImportError:
                _early_log(">>> GPU device_manager not available, skipping reset")

            # PRE-LOAD DATA FOR FORK COW SHARING
            # Loading data here (in parent) before fork means all workers
            # share the same memory pages via copy-on-write, saving significant memory:
            # - Headers: ~178MB per worker for 22M trace dataset
            # - Ensemble index: ~80MB per worker
            # Without this, each worker loads independently.

            # Pre-load ensemble index (all workers need this)
            ensemble_path = input_dir / 'ensemble_index.parquet'
            try:
                _early_log(f">>> Pre-loading ensemble index for fork COW sharing...")
                ensemble_df = read_parquet(ensemble_path)
                set_shared_ensemble_index(ensemble_df)
                _early_log(f">>> Ensemble index pre-loaded: {len(ensemble_df)} gathers, "
                          f"{ensemble_df.memory_usage(deep=True).sum()/(1024**2):.1f}MB "
                          f"(will be shared by all {self.n_workers} workers via COW)")
            except Exception as e:
                _early_log(f">>> WARNING: Failed to pre-load ensemble index: {e}, workers will load independently")

            # Pre-load headers if sorting, mute, or FKK processor is enabled
            mute_enabled = (
                self.config.mute_velocity > 0 and
                (self.config.mute_top or self.config.mute_bottom)
            )

            # Check if FKK processor is being used (needs ALL headers for volume building)
            processor_class = self.config.processor_config.get('class_name', '') if self.config.processor_config else ''
            is_fkk_processor = processor_class == 'FKKProcessor'

            header_columns = []
            load_all_headers = is_fkk_processor  # FKK needs all headers for inline/xline keys

            if not load_all_headers:
                if sorting_enabled:
                    header_columns.append(self.config.sort_options.sort_key)
                    if self.config.sort_options.secondary_key:
                        header_columns.append(self.config.sort_options.secondary_key)
                if mute_enabled:
                    for col in ['offset', 'OFFSET', 'Offset']:
                        if col not in header_columns:
                            header_columns.append(col)

            headers_path = input_dir / 'headers.parquet'
            if load_all_headers:
                # FKK processor needs ALL headers for volume building
                _early_log(f">>> FKK processor detected - pre-loading ALL headers for fork COW sharing")
                try:
                    shared_headers = read_parquet(headers_path)
                    all_cols = list(shared_headers.columns)
                    set_shared_headers(shared_headers, all_cols)
                    _early_log(f">>> ALL headers pre-loaded: {len(shared_headers)} rows, {len(all_cols)} columns, "
                              f"{shared_headers.memory_usage(deep=True).sum()/(1024**2):.1f}MB "
                              f"(will be shared by all {self.n_workers} workers via COW)")
                except Exception as e:
                    _early_log(f">>> WARNING: Failed to pre-load all headers: {e}, workers will load independently")
            elif header_columns:
                _early_log(f">>> Pre-loading headers for fork COW sharing: {header_columns}")
                try:
                    # Use parquet_io for case-insensitive column matching
                    col_mapping = find_columns_case_insensitive(headers_path, header_columns)
                    cols_to_load = [v for v in col_mapping.values() if v is not None]

                    if cols_to_load:
                        _early_log(f">>> Found columns to load: {cols_to_load}")
                        # Use Polars via parquet_io for 6x faster loading
                        shared_headers = read_parquet(headers_path, columns=cols_to_load)
                        set_shared_headers(shared_headers, cols_to_load)
                        _early_log(f">>> Headers pre-loaded: {len(shared_headers)} rows, "
                                  f"{shared_headers.memory_usage(deep=True).sum()/(1024**2):.1f}MB "
                                  f"(will be shared by all {self.n_workers} workers via COW)")
                    else:
                        _early_log(f">>> WARNING: None of requested columns {header_columns} found in parquet")
                except Exception as e:
                    _early_log(f">>> WARNING: Failed to pre-load headers: {e}, workers will load independently")

            # Log summary of pre-loaded shared data
            _early_log(f">>> {get_shared_data_summary()}")
        else:
            mp_ctx = mp.get_context('spawn')
            _early_log(">>> Using 'spawn' context (non-Linux platform)")
        _early_log(">>> Entering ProcessPoolExecutor context...")
        try:
            with ProcessPoolExecutor(max_workers=self.n_workers, mp_context=mp_ctx) as executor:
                self._executor = executor  # Store for cancellation
                _early_log(">>> ProcessPoolExecutor created, submitting tasks with staggered startup...")
                _debug_log("ProcessPoolExecutor created, submitting tasks with staggered startup...", include_memory=True)

                futures = {}
                # With fork context, workers share parent's imported modules via copy-on-write
                # No heavy import phase needed - workers start almost instantly
                # With spawn context, each worker needs ~3-5 seconds to complete imports
                if sys.platform == 'linux':
                    STAGGER_DELAY = 0.1  # Fork: workers share memory, fast startup
                else:
                    STAGGER_DELAY = 1.0  # Spawn: workers need time for imports

                # Check available memory before starting workers
                mem_before = psutil.virtual_memory()
                _early_log(f">>> Memory before workers: {mem_before.available/(1024**3):.2f}GB available")

                for i, task in enumerate(tasks):
                    mem = psutil.virtual_memory()
                    _early_log(f">>> Submitting task {i+1}/{len(tasks)} (segment {task.segment_id}) "
                              f"[{mem.available/(1024**3):.2f}GB avail]...")

                    # Check if we have enough memory for another worker
                    # Fork (Linux): Workers share imports via COW, need ~0.5GB for working data
                    # Spawn (other): Workers re-import everything, need ~2.5GB each
                    if sys.platform == 'linux':
                        MIN_MEMORY_GB = 1.0  # Fork: only need working memory
                    else:
                        MIN_MEMORY_GB = 3.0  # Spawn: need full import overhead
                    if mem.available / (1024**3) < MIN_MEMORY_GB:
                        _early_log(f">>> WARNING: Low memory! Only {mem.available/(1024**3):.2f}GB available, need {MIN_MEMORY_GB}GB")
                        # Wait for memory to free up (workers complete imports)
                        for wait_i in range(30):  # Wait up to 15 seconds
                            time.sleep(0.5)
                            mem = psutil.virtual_memory()
                            if mem.available / (1024**3) >= MIN_MEMORY_GB:
                                _early_log(f">>> Memory freed: {mem.available/(1024**3):.2f}GB available, continuing")
                                break
                            if wait_i % 4 == 0:  # Log every 2 seconds
                                _early_log(f">>> Waiting for memory... {mem.available/(1024**3):.2f}GB (need {MIN_MEMORY_GB}GB)")
                        else:
                            # Timeout - proceed anyway but log warning
                            _early_log(f">>> TIMEOUT waiting for memory, proceeding with {mem.available/(1024**3):.2f}GB")

                    future = executor.submit(process_gather_range, task, progress_queue)
                    futures[future] = task
                    self._futures = futures  # Store for cancellation
                    _early_log(f">>> Task {i+1} submitted, waiting {STAGGER_DELAY}s before next...")

                    # Stagger startup - give each worker time to complete heavy imports
                    # This prevents all workers from loading heavy modules simultaneously
                    if i < len(tasks) - 1:  # Don't sleep after last task
                        time.sleep(STAGGER_DELAY)

                _early_log(f">>> All {len(futures)} tasks submitted to executor (staggered)")
                _early_log(">>> About to call _debug_log for 'all tasks submitted'...")
                _debug_log(f"All {len(futures)} tasks submitted to executor (staggered)", include_memory=True)
                _early_log(">>> _debug_log done, setting start_time...")

                # Monitor progress
                start_time = time.time()
                _early_log(">>> start_time set, about to enter monitoring loop...")
                _early_log(">>> Entering monitoring loop...")
                _debug_log("Entering monitoring loop...", include_memory=True)

                loop_count = 0
                while len(processed_futures) < len(futures):
                    loop_count += 1
                    if loop_count % 10 == 1:  # Log every 10 iterations
                        mem = psutil.virtual_memory()
                        _early_log(f">>> Monitor loop #{loop_count}: {len(processed_futures)}/{len(futures)} done, "
                                  f"{mem.available/(1024**3):.2f}GB avail")

                    # Check completed futures
                    for future in futures:
                        if future.done() and future not in processed_futures:
                            processed_futures.add(future)
                            try:
                                result = future.result(timeout=0.1)
                                results.append(result)
                                worker_progress_traces[result.segment_id] = result.n_traces_processed
                                worker_progress_gathers[result.segment_id] = result.n_gathers_processed
                                print(f"    Worker {result.segment_id} completed: "
                                      f"{result.n_gathers_processed} gathers, "
                                      f"{result.n_traces_processed:,} traces in {result.elapsed_time:.1f}s")
                            except Exception as e:
                                _early_log(f">>> Worker error: {type(e).__name__}: {e}")
                                task = futures[future]
                                results.append(ProcessingWorkerResult(
                                    segment_id=task.segment_id,
                                    n_gathers_processed=0,
                                    n_traces_processed=0,
                                    elapsed_time=0,
                                    success=False,
                                    error=str(e)
                                ))

                    # Drain progress queue
                    while not progress_queue.empty():
                        try:
                            segment_id, traces_done, gathers_done = progress_queue.get_nowait()
                            worker_progress_traces[segment_id] = traces_done
                            worker_progress_gathers[segment_id] = gathers_done
                        except:
                            break

                    # Update progress callback
                    if progress_callback:
                        total_traces_done = sum(worker_progress_traces.values())
                        total_gathers_done = sum(worker_progress_gathers.values())
                        elapsed = time.time() - start_time
                        rate = total_traces_done / elapsed if elapsed > 0 else 0
                        eta = (n_traces - total_traces_done) / rate if rate > 0 else 0

                        try:
                            progress_callback(ProcessingProgress(
                                phase='processing',
                                current_traces=total_traces_done,
                                total_traces=n_traces,
                                current_gathers=total_gathers_done,
                                total_gathers=n_gathers,
                                active_workers=len(futures) - len(processed_futures),
                                worker_progress=worker_progress_traces.copy(),
                                elapsed_time=elapsed,
                                eta_seconds=eta
                            ))
                        except Exception as cb_err:
                            _early_log(f">>> ERROR in progress_callback: {type(cb_err).__name__}: {cb_err}")

                    # Sleep interval between UI updates (500ms to reduce blinking)
                    time.sleep(0.5)

        except Exception as executor_err:
            _early_log(f">>> EXECUTOR EXCEPTION: {type(executor_err).__name__}: {executor_err}")
            import traceback
            _early_log(f">>> TRACEBACK:\n{traceback.format_exc()}")
            raise
        finally:
            # Clean up shared data (headers + ensemble index) to free memory
            if sys.platform == 'linux':
                try:
                    clear_shared_data()
                    _early_log(">>> Cleared shared data (headers + ensemble index) from memory")
                except Exception:
                    pass

        return results

    def _create_sorted_headers(
        self,
        input_dir: Path,
        output_dir: Path,
        worker_results: List[ProcessingWorkerResult],
        ensemble_df: pd.DataFrame,
        n_traces: int
    ):
        """
        Create sorted headers.parquet from worker sort mappings.

        MEMORY OPTIMIZATIONS:
        - Uses streaming sort file format (reads incrementally)
        - Vectorized mapping construction (no Python loops)
        - Chunked header reordering (avoids full DataFrame copy)
        - Explicit garbage collection between phases

        Each worker saved sort mappings in streaming format.
        We load these, build a global reorder index, and create sorted headers.
        """
        logger.info(f"Creating sorted headers for {n_traces:,} traces...")

        # Phase 1: Build global sort mapping using vectorized operations
        print("    Building global sort mapping (vectorized)...")
        global_mapping = np.arange(n_traces, dtype=np.int64)  # Identity by default

        # Load and apply each worker's sort mappings
        for result in worker_results:
            if result.sort_mapping_path and Path(result.sort_mapping_path).exists():
                # Use streaming reader (handles both new binary and legacy pickle format)
                sort_mappings = read_streaming_sort_file(result.sort_mapping_path)

                # VECTORIZED: Apply each gather's sort mapping without Python loops
                for gather_idx, g_start, g_end, local_sort_indices in sort_mappings:
                    n_gather_traces = len(local_sort_indices)
                    if n_gather_traces > 0:
                        # Vectorized index computation - no inner loop!
                        old_global_positions = g_start + local_sort_indices
                        global_mapping[g_start:g_start + n_gather_traces] = old_global_positions

                # Free mappings memory immediately
                del sort_mappings
                gc.collect()

        # Phase 2: Chunked header reordering to avoid full DataFrame copy
        print("    Reordering headers (chunked)...")
        CHUNK_SIZE = 100_000  # Process 100k rows at a time

        headers_path = input_dir / 'headers.parquet'
        output_headers_path = output_dir / 'headers.parquet'

        # Load original headers
        headers_df = pd.read_parquet(headers_path)
        n_columns = len(headers_df.columns)
        logger.info(f"Headers loaded: {len(headers_df):,} rows, {n_columns} columns")

        # Process in chunks to avoid memory spike from full copy
        if n_traces > CHUNK_SIZE:
            # Large dataset: chunked processing
            chunks = []
            n_chunks = (n_traces + CHUNK_SIZE - 1) // CHUNK_SIZE

            for chunk_idx in range(n_chunks):
                chunk_start = chunk_idx * CHUNK_SIZE
                chunk_end = min(chunk_start + CHUNK_SIZE, n_traces)
                chunk_mapping = global_mapping[chunk_start:chunk_end]

                # Extract only the rows we need for this chunk
                chunk_headers = headers_df.iloc[chunk_mapping].copy()
                chunk_headers['trace_index'] = np.arange(chunk_start, chunk_end)
                chunk_headers['original_trace_index'] = chunk_mapping

                chunks.append(chunk_headers)

                # Progress update for large datasets
                if (chunk_idx + 1) % 10 == 0 or chunk_idx == n_chunks - 1:
                    logger.debug(f"Header reorder: chunk {chunk_idx + 1}/{n_chunks}")

                # Cleanup
                del chunk_mapping
                if chunk_idx % 5 == 0:
                    gc.collect()

            # Concatenate chunks
            sorted_headers = pd.concat(chunks, ignore_index=True)
            del chunks
            gc.collect()
        else:
            # Small dataset: direct reorder is OK
            sorted_headers = headers_df.iloc[global_mapping].reset_index(drop=True)
            sorted_headers['trace_index'] = np.arange(len(sorted_headers))
            sorted_headers['original_trace_index'] = global_mapping

        # Free original headers memory
        del headers_df
        del global_mapping
        gc.collect()

        # Phase 3: Save sorted headers
        print("    Saving sorted headers...")
        sorted_headers.to_parquet(output_headers_path)

        n_saved = len(sorted_headers)
        del sorted_headers
        gc.collect()

        # Copy ensemble index (gather boundaries remain the same, just internal order changed)
        ensemble_src = input_dir / 'ensemble_index.parquet'
        ensemble_dst = output_dir / 'ensemble_index.parquet'
        if not ensemble_src.exists():
            raise FileNotFoundError(
                f"Required file missing: ensemble_index.parquet. "
                f"Input dataset at {input_dir} may be corrupted."
            )
        try:
            shutil.copy2(ensemble_src, ensemble_dst)
            if not ensemble_dst.exists():
                raise RuntimeError("ensemble_index.parquet copy failed")
        except Exception as e:
            raise RuntimeError(f"Failed to copy ensemble_index.parquet: {e}") from e

        # Copy trace index if exists (optional)
        trace_idx_src = input_dir / 'trace_index.parquet'
        if trace_idx_src.exists():
            try:
                shutil.copy2(trace_idx_src, output_dir / 'trace_index.parquet')
            except Exception as e:
                logger.warning(f"Failed to copy optional trace_index.parquet: {e}")

        print(f"    Sorted headers saved: {n_saved:,} traces")
        logger.info(f"Sorted headers complete: {n_saved:,} traces")

    def _save_processing_metadata(
        self,
        output_dir: Path,
        original_metadata: dict,
        n_workers: int,
        sorting_enabled: bool
    ):
        """Save processing metadata, preserving original SEG-Y path for export."""
        metadata = original_metadata.copy()
        processing_info = {
            'method': 'parallel_multiprocess',
            'n_workers': n_workers,
            'processor_config': self.config.processor_config
        }

        if sorting_enabled and self.config.sort_options:
            processing_info['sorting'] = {
                'enabled': True,
                'sort_key': self.config.sort_options.sort_key,
                'ascending': self.config.sort_options.ascending,
                'secondary_key': self.config.sort_options.secondary_key,
                'secondary_ascending': self.config.sort_options.secondary_ascending
            }

        metadata['processing_info'] = processing_info

        # CRITICAL: Preserve original_segy_path for export functionality
        # The path may be at top level or nested in seismic_metadata
        original_segy_path = None

        # Check top-level first
        if 'original_segy_path' in original_metadata:
            original_segy_path = original_metadata['original_segy_path']

        # Fall back to seismic_metadata section
        if not original_segy_path:
            seismic_meta = original_metadata.get('seismic_metadata', {})
            original_segy_path = seismic_meta.get('original_segy_path') or seismic_meta.get('source_file')

        # Ensure path is at top level for export to find it
        if original_segy_path:
            metadata['original_segy_path'] = original_segy_path
            # Also preserve in seismic_metadata for consistency
            if 'seismic_metadata' not in metadata:
                metadata['seismic_metadata'] = {}
            metadata['seismic_metadata']['original_segy_path'] = original_segy_path
            logger.info(f"Preserved original SEG-Y path: {original_segy_path}")
        else:
            logger.warning(
                "No original_segy_path found in input metadata. "
                "Export to SEG-Y may not work. Re-import original SEG-Y to fix."
            )

        metadata_path = output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _validate_output_files(self, output_dir: Path, output_mode: str) -> None:
        """Validate all required output files exist after processing.

        Parameters
        ----------
        output_dir : Path
            Output directory to validate
        output_mode : str
            Output mode ('processed', 'noise', 'both')

        Raises
        ------
        RuntimeError
            If required output files are missing
        """
        required_files = [
            'headers.parquet',
            'ensemble_index.parquet',
            'metadata.json',
        ]

        # Add appropriate zarr file based on output mode
        if output_mode == 'noise':
            required_files.append('noise.zarr')
            # traces.zarr should be a symlink to noise.zarr
        else:
            required_files.append('traces.zarr')

        if output_mode == 'both':
            required_files.append('noise.zarr')

        missing_files = []
        for filename in required_files:
            path = output_dir / filename
            if not path.exists():
                missing_files.append(filename)

        if missing_files:
            raise RuntimeError(
                f"Output validation failed. Missing files in {output_dir}: "
                f"{', '.join(missing_files)}. "
                f"Processing may have failed silently. Check logs for details."
            )

        _debug_log(f"Output validation passed: all {len(required_files)} required files present")

    def cancel(self):
        """Request cancellation of processing."""
        self._cancel_requested = True
        self._force_terminate_workers()

    def _force_terminate_workers(self):
        """Force terminate all worker processes."""
        _early_log(">>> CANCEL: Force terminating worker processes...")

        # Cancel pending futures
        if self._futures:
            for future in self._futures:
                if not future.done():
                    future.cancel()

        # Shutdown executor and terminate processes
        if self._executor:
            try:
                # Shutdown without waiting
                self._executor.shutdown(wait=False, cancel_futures=True)

                # Force terminate any processes still running
                if hasattr(self._executor, '_processes'):
                    for pid, process in list(self._executor._processes.items()):
                        try:
                            _early_log(f">>> CANCEL: Terminating worker process {pid}")
                            process.terminate()
                        except Exception as e:
                            _early_log(f">>> CANCEL: Error terminating {pid}: {e}")
                            try:
                                process.kill()
                            except Exception:
                                pass
            except Exception as e:
                _early_log(f">>> CANCEL: Error during shutdown: {e}")

        _early_log(">>> CANCEL: Worker termination complete")

    def _check_memory_budget(
        self,
        n_workers: int,
        max_gather_traces: int,
        n_samples: int,
        sorting_enabled: bool = False,
        output_noise: bool = False,
        output_mode: str = 'processed'
    ) -> Optional[Tuple[bool, float, float, float, str, str]]:
        """
        Check memory budget for parallel processing.

        Uses settings from AppSettings if available, otherwise uses defaults.

        Args:
            n_workers: Number of parallel workers
            max_gather_traces: Maximum traces in any single gather
            n_samples: Samples per trace
            sorting_enabled: Whether sorting is enabled
            output_noise: Whether noise output is enabled (legacy)
            output_mode: 'processed', 'noise', or 'both'

        Returns:
            Tuple of (is_safe, available_mb, required_mb, ratio, risk_level, message)
            or None if check is disabled
        """
        # Get settings
        if SETTINGS_AVAILABLE:
            settings = get_settings()
            safety_factor = settings.get_parallel_memory_safety_factor() / 100.0
            memory_copies = settings.get_parallel_memory_copies_estimate()
            warn_enabled = settings.get_parallel_warn_on_memory_risk()
        else:
            # Defaults if settings not available
            safety_factor = 0.7
            memory_copies = 4
            warn_enabled = True

        if not warn_enabled:
            return None

        return check_parallel_processing_memory_budget(
            n_workers=n_workers,
            max_gather_traces=max_gather_traces,
            n_samples=n_samples,
            memory_copies=memory_copies,
            safety_factor=safety_factor,
            output_noise=output_noise,
            sorting_enabled=sorting_enabled,
            output_mode=output_mode
        )
