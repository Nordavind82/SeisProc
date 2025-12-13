"""
QC Batch Engine - Process selected gathers through a processing chain

Handles:
1. Loading velocity model (optional NMO)
2. Identifying contributing gathers for selected inlines
3. Applying processing chain to each gather
4. Generating before/after stacks
5. Computing difference volumes
6. Progress reporting

Usage:
    from processors.qc_batch_engine import QCBatchEngine
    from views.qc_batch_dialog import QCBatchConfig

    engine = QCBatchEngine(config)
    engine.progress_updated.connect(on_progress)
    engine.run()
"""

import numpy as np
import zarr
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass
import json
import logging

from PyQt6.QtCore import QObject, pyqtSignal, QThread

from models.velocity_model import VelocityModel
from processors.base_processor import BaseProcessor
from processors.nmo_processor import NMOProcessor, NMOConfig
from processors.cdp_stacker import CDPStacker, StackConfig, StackResult

logger = logging.getLogger(__name__)


def apply_mute_to_trace(
    trace: np.ndarray,
    offset: float,
    sample_interval_ms: float,
    velocity: float,
    top_mute: bool,
    bottom_mute: bool,
    taper_samples: int
) -> np.ndarray:
    """
    Apply linear mute to a single trace.

    Mute formula: T = |offset| / velocity

    Args:
        trace: Trace data array (modified in place)
        offset: Offset in meters
        sample_interval_ms: Sample interval in milliseconds
        velocity: Mute velocity in m/s
        top_mute: Zero samples before mute time
        bottom_mute: Zero samples after mute time
        taper_samples: Number of samples for cosine taper

    Returns:
        Modified trace
    """
    n_samples = len(trace)

    # Calculate mute sample
    velocity_m_per_ms = velocity / 1000.0
    mute_time_ms = abs(offset) / velocity_m_per_ms
    mute_sample = int(mute_time_ms / sample_interval_ms)

    # Pre-compute taper if needed
    if taper_samples > 0:
        taper = 0.5 * (1 - np.cos(np.linspace(0, np.pi, taper_samples)))
    else:
        taper = np.array([])

    # Apply top mute
    if top_mute and mute_sample > 0:
        mute_end = min(mute_sample, n_samples)
        trace[:mute_end] = 0

        # Apply taper after mute zone
        if taper_samples > 0 and mute_end < n_samples:
            taper_end = min(mute_end + taper_samples, n_samples)
            actual_taper_len = taper_end - mute_end
            if actual_taper_len > 0:
                trace[mute_end:taper_end] *= taper[:actual_taper_len]

    # Apply bottom mute
    if bottom_mute and mute_sample < n_samples:
        mute_start = max(0, mute_sample)

        # Apply taper before mute zone
        if taper_samples > 0 and mute_start > 0:
            taper_start = max(0, mute_start - taper_samples)
            actual_taper_len = mute_start - taper_start
            if actual_taper_len > 0:
                trace[taper_start:mute_start] *= taper[:actual_taper_len][::-1]

        # Zero after mute
        trace[mute_start:] = 0

    return trace


def apply_mute_to_gather(
    traces: np.ndarray,
    offsets: np.ndarray,
    sample_interval_ms: float,
    velocity: float,
    top_mute: bool,
    bottom_mute: bool,
    taper_samples: int
) -> np.ndarray:
    """
    Apply linear mute to all traces in a gather.

    Args:
        traces: Trace data array (n_samples, n_traces)
        offsets: Offset values for each trace
        sample_interval_ms: Sample interval in milliseconds
        velocity: Mute velocity in m/s
        top_mute: Zero samples before mute time
        bottom_mute: Zero samples after mute time
        taper_samples: Number of samples for cosine taper

    Returns:
        Muted traces
    """
    result = traces.copy()
    for i in range(traces.shape[1]):
        result[:, i] = apply_mute_to_trace(
            result[:, i].copy(),
            offsets[i],
            sample_interval_ms,
            velocity,
            top_mute,
            bottom_mute,
            taper_samples
        )
    return result


@dataclass
class BatchProgress:
    """Progress information for batch processing workflow."""
    phase: str  # 'initializing', 'processing', 'stacking', 'writing', 'complete'
    current: int
    total: int
    message: str
    inline_current: Optional[int] = None
    gather_current: Optional[int] = None
    percent: float = 0.0


@dataclass
class BatchResult:
    """Result from batch processing."""
    success: bool
    output_dir: str
    output_files: Dict[str, str]  # name -> path
    stats: Dict[str, Any]
    error_message: Optional[str] = None


class QCBatchEngine(QObject):
    """
    Engine for executing QC batch processing workflow.

    Processes selected gathers through a configurable processing chain,
    optionally applies NMO, and generates before/after stacks with
    difference volumes for QC.
    """

    progress_updated = pyqtSignal(object)  # BatchProgress
    batch_complete = pyqtSignal(object)    # BatchResult
    batch_error = pyqtSignal(str)          # Error message

    def __init__(self, config: 'QCBatchConfig', parent=None):
        """
        Initialize batch engine.

        Args:
            config: QCBatchConfig from dialog
            parent: Optional QObject parent
        """
        super().__init__(parent)
        self.config = config
        self._velocity_model: Optional[VelocityModel] = None
        self._nmo_processor: Optional[NMOProcessor] = None
        self._processors: List[BaseProcessor] = []
        self._cancelled = False

    def cancel(self):
        """Request cancellation of running operation."""
        self._cancelled = True

    def run(self) -> Optional[BatchResult]:
        """
        Execute the full batch processing workflow.

        Returns:
            BatchResult on success, None on failure
        """
        self._cancelled = False

        try:
            # Phase 1: Initialize
            self._report_progress('initializing', 0, 4, "Initializing...")
            self._initialize()

            if self._cancelled:
                return None

            # Phase 2: Process gathers
            self._report_progress('processing', 1, 4, "Processing gathers...")
            before_data, after_data, noise_data, gather_info = self._process_gathers()

            if self._cancelled:
                return None

            # Phase 3: Stack
            self._report_progress('stacking', 2, 4, "Stacking...")
            stacks = self._generate_stacks(before_data, after_data, noise_data, gather_info)

            if self._cancelled:
                return None

            # Phase 4: Write outputs
            self._report_progress('writing', 3, 4, "Writing outputs...")
            result = self._write_outputs(before_data, after_data, noise_data, stacks, gather_info)

            self._report_progress('complete', 4, 4, "Complete!")
            self.batch_complete.emit(result)

            return result

        except Exception as e:
            logger.exception("QC batch processing failed")
            self.batch_error.emit(str(e))
            return None

    def _report_progress(
        self,
        phase: str,
        current: int,
        total: int,
        message: str,
        inline: Optional[int] = None,
        gather: Optional[int] = None
    ):
        """Emit progress signal."""
        percent = (current / total * 100) if total > 0 else 0
        progress = BatchProgress(
            phase=phase,
            current=current,
            total=total,
            message=message,
            inline_current=inline,
            gather_current=gather,
            percent=percent
        )
        self.progress_updated.emit(progress)

    def _initialize(self):
        """Initialize components: velocity model, NMO, processing chain."""
        # Initialize processing chain
        self._processors = self._create_processor_chain()
        logger.info(f"Initialized processing chain with {len(self._processors)} processors")

        # Initialize NMO if requested
        if self.config.apply_nmo:
            self._load_velocity_model()
            nmo_config = NMOConfig(
                stretch_mute_factor=self.config.stretch_mute,
                velocity_type=self.config.velocity_type,
            )
            self._nmo_processor = NMOProcessor(nmo_config, self._velocity_model)
            logger.info("Initialized NMO processor")

    def _create_processor_chain(self) -> List[BaseProcessor]:
        """Create processor instances from chain configuration."""
        from processors import (
            BandpassFilter, FKFilter, GainProcessor,
            TFDenoise, STFTDenoise, DWTDenoise
        )

        processor_classes = {
            'bandpass': BandpassFilter,
            'fk_filter': FKFilter,
            'gain': GainProcessor,
            'tf_denoise': TFDenoise,
            'stft_denoise': STFTDenoise,
            'dwt_denoise': DWTDenoise,
        }

        processors = []
        for config in self.config.processing_chain:
            name = config.get('name')
            params = config.get('params', {})

            if name in processor_classes:
                try:
                    proc = processor_classes[name](**params)
                    processors.append(proc)
                except Exception as e:
                    logger.warning(f"Failed to create processor {name}: {e}")

        return processors

    def _load_velocity_model(self):
        """Load velocity model from configured file."""
        from utils.velocity_io import preview_velocity_file, velocity_info_to_model

        info = preview_velocity_file(self.config.velocity_file)

        if not info.is_valid:
            raise ValueError(f"Failed to read velocity file: {info.error_message}")

        self._velocity_model = velocity_info_to_model(
            info,
            velocity_type=self.config.velocity_type,
            time_unit=self.config.time_unit,
            velocity_unit=self.config.velocity_unit,
        )

        logger.info(f"Loaded velocity model: {self._velocity_model}")

    def _identify_gathers(self) -> Dict[int, List[Dict[str, Any]]]:
        """
        Identify CDP gathers contributing to selected inlines.

        Returns:
            Dictionary mapping inline_no -> list of gather info dicts
        """
        dataset_path = Path(self.config.dataset_path)
        ensemble_path = dataset_path / "ensemble_index.parquet"

        if not ensemble_path.exists():
            raise FileNotFoundError(f"Ensemble index not found: {ensemble_path}")

        # Load ensemble index
        df = pd.read_parquet(ensemble_path)

        # Find inline column
        inline_col = None
        for col in ['INLINE_NO', 'inline_no', 'Inline', 'INLINE']:
            if col in df.columns:
                inline_col = col
                break

        if inline_col is None:
            raise ValueError("No inline column found in ensemble index")

        # Build gather info by inline
        gather_info = {}
        for inline in self.config.inline_numbers:
            mask = df[inline_col] == inline
            inline_df = df[mask]

            gathers = []
            for _, row in inline_df.iterrows():
                gathers.append({
                    'ensemble_id': row.get('ensemble_id', row.name),
                    'start_trace': row['start_trace'],
                    'end_trace': row['end_trace'],
                    'n_traces': row.get('n_traces', row['end_trace'] - row['start_trace']),
                    'cdp': row.get('CDP', row.get('cdp', row.get('ensemble_id', 0))),
                    'inline': inline,
                })
            gather_info[inline] = gathers

        return gather_info

    def _process_gathers(self) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Process all gathers through the processing chain.

        Returns:
            Tuple of (before_data, after_data, noise_data, gather_info)
        """
        dataset_path = Path(self.config.dataset_path)

        # Open data stores
        traces_zarr = zarr.open(dataset_path / "traces.zarr", mode='r')
        headers_df = pd.read_parquet(dataset_path / "headers.parquet")

        n_samples = traces_zarr.shape[0]
        sample_interval_ms = self._get_sample_interval(dataset_path)

        # Identify gathers
        gather_info = self._identify_gathers()
        total_gathers = sum(len(g) for g in gather_info.values())

        # Storage for before/after/noise data
        before_data = {}  # (inline, cdp) -> traces
        after_data = {}   # (inline, cdp) -> traces
        noise_data = {}   # (inline, cdp) -> traces (before - after)

        # Check if mute is enabled
        apply_mute = (
            self.config.mute_velocity > 0 and
            (self.config.mute_top or self.config.mute_bottom)
        )

        gather_idx = 0
        for inline in self.config.inline_numbers:
            if self._cancelled:
                break

            gathers = gather_info.get(inline, [])

            for gather in gathers:
                if self._cancelled:
                    break

                self._report_progress(
                    'processing',
                    gather_idx,
                    total_gathers,
                    f"Inline {inline}, CDP {gather['cdp']}",
                    inline=inline,
                    gather=gather['cdp']
                )

                # Load gather traces
                start = gather['start_trace']
                end = gather['end_trace']
                traces = np.array(traces_zarr[:, start:end], dtype=np.float32)

                # Get offsets
                gather_headers = headers_df.iloc[start:end]
                offset_col = None
                for col in ['OFFSET', 'offset', 'Offset']:
                    if col in gather_headers.columns:
                        offset_col = col
                        break

                if offset_col:
                    offsets = gather_headers[offset_col].values.astype(np.float32)
                else:
                    offsets = np.arange(end - start, dtype=np.float32) * 100

                # Apply NMO if configured
                if self._nmo_processor is not None:
                    before_traces = self._nmo_processor.apply_nmo(
                        traces, offsets, sample_interval_ms, cdp=gather['cdp']
                    )
                else:
                    before_traces = traces.copy()

                # Store before data
                key = (inline, gather['cdp'])
                before_data[key] = before_traces

                # Apply processing chain
                after_traces = before_traces.copy()
                for processor in self._processors:
                    try:
                        after_traces = processor.process(after_traces, sample_interval_ms)
                    except Exception as e:
                        logger.warning(f"Processor {processor.__class__.__name__} failed: {e}")

                # Store after data
                after_data[key] = after_traces

                # Calculate noise (before - after = input - processed) with optional mute
                if apply_mute:
                    # Apply mute based on target
                    if self.config.mute_target == 'input':
                        # Mute applied to input before subtraction
                        muted_before = apply_mute_to_gather(
                            before_traces, offsets, sample_interval_ms,
                            self.config.mute_velocity,
                            self.config.mute_top, self.config.mute_bottom,
                            self.config.mute_taper
                        )
                        noise_traces = muted_before - after_traces
                    elif self.config.mute_target == 'processed':
                        # Mute applied to processed before subtraction
                        muted_after = apply_mute_to_gather(
                            after_traces, offsets, sample_interval_ms,
                            self.config.mute_velocity,
                            self.config.mute_top, self.config.mute_bottom,
                            self.config.mute_taper
                        )
                        noise_traces = before_traces - muted_after
                    else:  # 'output' - apply mute to noise result
                        noise_traces = before_traces - after_traces
                        noise_traces = apply_mute_to_gather(
                            noise_traces, offsets, sample_interval_ms,
                            self.config.mute_velocity,
                            self.config.mute_top, self.config.mute_bottom,
                            self.config.mute_taper
                        )
                else:
                    # No mute - simple subtraction
                    noise_traces = before_traces - after_traces

                noise_data[key] = noise_traces

                # Store metadata in gather info
                gather['offsets'] = offsets
                gather['n_samples'] = n_samples
                gather['sample_interval_ms'] = sample_interval_ms

                gather_idx += 1

        return before_data, after_data, noise_data, gather_info

    def _generate_stacks(
        self,
        before_data: Dict,
        after_data: Dict,
        noise_data: Dict,
        gather_info: Dict
    ) -> Dict[str, np.ndarray]:
        """
        Generate stacks from processed gathers.

        Returns:
            Dictionary with 'before_stack', 'after_stack', 'noise_stack', 'difference'
        """
        stack_config = StackConfig(
            method=self.config.stack_method,
            min_fold=self.config.min_fold,
        )
        stacker = CDPStacker(stack_config)

        # Get dimensions from first gather
        first_key = list(before_data.keys())[0] if before_data else None
        if first_key is None:
            return {}

        n_samples = before_data[first_key].shape[0]
        n_cdps = len(before_data)

        # Allocate stack arrays
        before_stack = np.zeros((n_samples, n_cdps), dtype=np.float32)
        after_stack = np.zeros((n_samples, n_cdps), dtype=np.float32)
        noise_stack = np.zeros((n_samples, n_cdps), dtype=np.float32)
        fold_data = np.zeros((n_samples, n_cdps), dtype=np.int32)

        # Stack each CDP
        for idx, key in enumerate(sorted(before_data.keys())):
            before_traces = before_data[key]
            after_traces = after_data[key]
            noise_traces = noise_data.get(key)

            # Stack before
            before_result = stacker.stack_gather(before_traces)
            before_stack[:, idx] = before_result.trace

            # Stack after
            after_result = stacker.stack_gather(after_traces)
            after_stack[:, idx] = after_result.trace

            # Stack noise
            if noise_traces is not None:
                noise_result = stacker.stack_gather(noise_traces)
                noise_stack[:, idx] = noise_result.trace

            fold_data[:, idx] = before_result.fold

        # Compute difference (after_stack - before_stack)
        difference = after_stack - before_stack

        return {
            'before_stack': before_stack,
            'after_stack': after_stack,
            'noise_stack': noise_stack,
            'difference': difference,
            'fold': fold_data,
            'cdp_keys': sorted(before_data.keys()),
        }

    def _get_sample_interval(self, dataset_path: Path) -> float:
        """Get sample interval in milliseconds from metadata."""
        metadata_path = dataset_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            dt = metadata.get('sample_interval', 0.004)
            return dt * 1000 if dt < 1 else dt
        return 4.0

    def _write_outputs(
        self,
        before_data: Dict,
        after_data: Dict,
        noise_data: Dict,
        stacks: Dict,
        gather_info: Dict
    ) -> BatchResult:
        """
        Write output files based on configuration.

        Returns:
            BatchResult with output information
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_name = self.config.output_name
        output_files = {}
        stats = {
            'n_inlines': len(self.config.inline_numbers),
            'n_gathers': len(before_data),
            'n_processors': len(self._processors),
        }

        # Get sample info from first gather
        first_key = list(before_data.keys())[0] if before_data else None
        if first_key is None:
            return BatchResult(
                success=False,
                output_dir=str(output_dir),
                output_files={},
                stats=stats,
                error_message="No gathers processed"
            )

        n_samples = before_data[first_key].shape[0]
        dataset_path = Path(self.config.dataset_path)
        sample_interval_ms = self._get_sample_interval(dataset_path)

        # Write before gathers
        if self.config.output_before_gathers:
            path = self._write_gather_zarr(
                output_dir / f"{output_name}_before_gathers.zarr",
                before_data
            )
            output_files['before_gathers'] = str(path)

        # Write after gathers
        if self.config.output_after_gathers:
            path = self._write_gather_zarr(
                output_dir / f"{output_name}_after_gathers.zarr",
                after_data
            )
            output_files['after_gathers'] = str(path)

        # Write noise gathers
        if self.config.output_noise_gathers:
            path = self._write_gather_zarr(
                output_dir / f"{output_name}_noise_gathers.zarr",
                noise_data
            )
            output_files['noise_gathers'] = str(path)

        # Write before stack
        if self.config.output_before_stack and 'before_stack' in stacks:
            path = self._write_stack_zarr(
                output_dir / f"{output_name}_before_stack.zarr",
                stacks['before_stack']
            )
            output_files['before_stack'] = str(path)

        # Write after stack
        if self.config.output_after_stack and 'after_stack' in stacks:
            path = self._write_stack_zarr(
                output_dir / f"{output_name}_after_stack.zarr",
                stacks['after_stack']
            )
            output_files['after_stack'] = str(path)

        # Write noise stack
        if self.config.output_noise_stack and 'noise_stack' in stacks:
            path = self._write_stack_zarr(
                output_dir / f"{output_name}_noise_stack.zarr",
                stacks['noise_stack']
            )
            output_files['noise_stack'] = str(path)

        # Write difference
        if self.config.output_difference and 'difference' in stacks:
            path = self._write_stack_zarr(
                output_dir / f"{output_name}_difference.zarr",
                stacks['difference']
            )
            output_files['difference'] = str(path)

        # Write metadata
        metadata = {
            'type': 'qc_batch',
            'source_dataset': str(self.config.dataset_path),
            'inline_numbers': self.config.inline_numbers,
            'processing_chain': self.config.processing_chain,
            'apply_nmo': self.config.apply_nmo,
            'velocity_file': self.config.velocity_file if self.config.apply_nmo else None,
            'stretch_mute': self.config.stretch_mute if self.config.apply_nmo else None,
            'stack_method': self.config.stack_method,
            'min_fold': self.config.min_fold,
            'mute_velocity': self.config.mute_velocity,
            'mute_top': self.config.mute_top,
            'mute_bottom': self.config.mute_bottom,
            'mute_taper': self.config.mute_taper,
            'mute_target': self.config.mute_target,
            'n_samples': n_samples,
            'n_cdps': len(before_data),
            'cdp_keys': [list(k) for k in stacks.get('cdp_keys', [])],
            'sample_interval': sample_interval_ms / 1000,
            'output_files': output_files,
        }

        metadata_path = output_dir / f"{output_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        output_files['metadata'] = str(metadata_path)

        # Compute statistics
        if 'before_stack' in stacks and 'after_stack' in stacks:
            before_rms = np.sqrt(np.mean(stacks['before_stack'] ** 2))
            after_rms = np.sqrt(np.mean(stacks['after_stack'] ** 2))
            diff_rms = np.sqrt(np.mean(stacks['difference'] ** 2))
            noise_rms = np.sqrt(np.mean(stacks['noise_stack'] ** 2)) if 'noise_stack' in stacks else 0.0

            # Correlation
            before_flat = stacks['before_stack'].flatten()
            after_flat = stacks['after_stack'].flatten()
            correlation = np.corrcoef(before_flat, after_flat)[0, 1]

            stats.update({
                'before_rms': float(before_rms),
                'after_rms': float(after_rms),
                'noise_rms': float(noise_rms),
                'difference_rms': float(diff_rms),
                'correlation': float(correlation),
            })

        logger.info(f"Wrote QC batch outputs to {output_dir}")

        return BatchResult(
            success=True,
            output_dir=str(output_dir),
            output_files=output_files,
            stats=stats
        )

    def _write_gather_zarr(self, path: Path, gather_data: Dict) -> Path:
        """Write gathers to Zarr store."""
        # Concatenate all gathers
        all_traces = []
        for key in sorted(gather_data.keys()):
            traces = gather_data[key]
            all_traces.append(traces)

        if all_traces:
            combined = np.concatenate(all_traces, axis=1)
        else:
            combined = np.zeros((0, 0), dtype=np.float32)

        # Write Zarr
        store = zarr.open(
            str(path),
            mode='w',
            shape=combined.shape,
            dtype=np.float32,
            chunks=(combined.shape[0], min(64, combined.shape[1]))
        )
        store[:] = combined

        return path

    def _write_stack_zarr(self, path: Path, stack_data: np.ndarray) -> Path:
        """Write stack to Zarr store."""
        store = zarr.open(
            str(path),
            mode='w',
            shape=stack_data.shape,
            dtype=np.float32,
            chunks=(stack_data.shape[0], min(64, stack_data.shape[1]))
        )
        store[:] = stack_data

        return path


# =============================================================================
# Worker Thread for Background Execution
# =============================================================================

class QCBatchWorker(QThread):
    """
    Worker thread for running QC batch processing in background.

    Usage:
        worker = QCBatchWorker(config)
        worker.progress_updated.connect(on_progress)
        worker.finished_with_result.connect(on_complete)
        worker.start()
    """

    progress_updated = pyqtSignal(object)  # BatchProgress
    finished_with_result = pyqtSignal(object)  # BatchResult
    error_occurred = pyqtSignal(str)

    def __init__(self, config: 'QCBatchConfig', parent=None):
        super().__init__(parent)
        self.config = config
        self._engine: Optional[QCBatchEngine] = None

    def run(self):
        """Execute batch processing in thread."""
        self._engine = QCBatchEngine(self.config)
        self._engine.progress_updated.connect(self.progress_updated.emit)

        try:
            result = self._engine.run()
            if result and result.success:
                self.finished_with_result.emit(result)
            else:
                self.error_occurred.emit(
                    result.error_message if result else "Processing cancelled or failed"
                )
        except Exception as e:
            logger.exception("Batch worker error")
            self.error_occurred.emit(str(e))

    def cancel(self):
        """Request cancellation."""
        if self._engine:
            self._engine.cancel()
