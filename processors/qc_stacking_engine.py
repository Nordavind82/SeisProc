"""
QC Stacking Engine - Orchestrates the QC stacking workflow

Handles:
1. Loading velocity model
2. Identifying contributing gathers for selected inlines
3. NMO correction and stacking for each CDP
4. Assembly into output stacked sections
5. Progress reporting

Usage:
    from processors.qc_stacking_engine import QCStackingEngine
    from views.qc_stacking_dialog import QCStackingConfig

    engine = QCStackingEngine(config)
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

from PyQt6.QtCore import QObject, pyqtSignal

from models.velocity_model import VelocityModel
from utils.velocity_io import read_velocity_auto, velocity_info_to_model, preview_velocity_file
from processors.nmo_processor import NMOProcessor, NMOConfig
from processors.cdp_stacker import CDPStacker, StackConfig, StackResult

logger = logging.getLogger(__name__)


@dataclass
class StackingProgress:
    """Progress information for stacking workflow."""
    phase: str  # 'loading', 'identifying', 'stacking', 'writing', 'complete'
    current: int
    total: int
    message: str
    inline_current: Optional[int] = None
    cdp_current: Optional[int] = None


class QCStackingEngine(QObject):
    """
    Engine for executing QC stacking workflow.

    Emits progress signals during execution for UI updates.
    """

    progress_updated = pyqtSignal(object)  # StackingProgress
    stacking_complete = pyqtSignal(str)    # Output path
    stacking_error = pyqtSignal(str)       # Error message

    def __init__(self, config: 'QCStackingConfig', parent=None):
        """
        Initialize stacking engine.

        Args:
            config: QCStackingConfig from dialog
            parent: Optional QObject parent
        """
        super().__init__(parent)
        self.config = config
        self._velocity_model: Optional[VelocityModel] = None
        self._cancelled = False

    def cancel(self):
        """Request cancellation of running operation."""
        self._cancelled = True

    def run(self) -> Optional[str]:
        """
        Execute the full stacking workflow.

        Returns:
            Output path on success, None on failure
        """
        self._cancelled = False

        try:
            # Phase 1: Load velocity model
            self._report_progress('loading', 0, 3, "Loading velocity model...")
            self._load_velocity_model()

            if self._cancelled:
                return None

            # Phase 2: Identify contributing gathers
            self._report_progress('identifying', 1, 3, "Identifying gathers...")
            gather_info = self._identify_gathers()

            if self._cancelled:
                return None

            # Phase 3: Stack each inline
            self._report_progress('stacking', 2, 3, "Stacking...")
            stacked_data, fold_data, cdp_axis, inline_ranges = self._stack_inlines(gather_info)

            if self._cancelled:
                return None

            # Phase 4: Write output
            self._report_progress('writing', 3, 3, "Writing output...")
            output_path = self._write_output(stacked_data, fold_data, cdp_axis, inline_ranges)

            self._report_progress('complete', 3, 3, "Complete!")
            self.stacking_complete.emit(output_path)

            return output_path

        except Exception as e:
            logger.exception("QC stacking failed")
            self.stacking_error.emit(str(e))
            return None

    def _report_progress(
        self,
        phase: str,
        current: int,
        total: int,
        message: str,
        inline: Optional[int] = None,
        cdp: Optional[int] = None
    ):
        """Emit progress signal."""
        progress = StackingProgress(
            phase=phase,
            current=current,
            total=total,
            message=message,
            inline_current=inline,
            cdp_current=cdp
        )
        self.progress_updated.emit(progress)

    def _load_velocity_model(self):
        """Load velocity model from configured file and optionally interpolate to grid."""
        # Build custom header mapping for SEG-Y velocity files if configured
        inline_byte = None
        xline_byte = None
        if getattr(self.config, 'use_custom_vel_bytes', False):
            inline_byte = self.config.vel_inline_byte
            xline_byte = self.config.vel_xline_byte
            logger.info(f"Using custom velocity SEG-Y bytes: inline={inline_byte}, xline={xline_byte}")

        # Preview velocity file with custom byte positions if specified
        info = preview_velocity_file(
            self.config.velocity_file,
            inline_byte=inline_byte,
            xline_byte=xline_byte
        )

        if not info.is_valid:
            raise ValueError(f"Failed to read velocity file: {info.error_message}")

        # Log detected velocity spatial info
        logger.info(
            f"Velocity file spatial info - Inline: {info.inline_range}, "
            f"Xline: {info.xline_range}, CDP: {info.cdp_range}"
        )

        # Store custom byte mapping in info for later use
        if inline_byte is not None:
            info.raw_data = info.raw_data or {}
            info.raw_data['custom_header_mapping'] = {
                'inline_byte': inline_byte,
                'xline_byte': xline_byte,
            }

        self._velocity_model = velocity_info_to_model(
            info,
            velocity_type=self.config.velocity_type,
            time_unit=self.config.time_unit,
            velocity_unit=self.config.velocity_unit,
        )

        logger.info(f"Loaded velocity model: {self._velocity_model}")

        # Update config with detected velocity spatial info (for extrapolation)
        # These will be used in _interpolate_velocity_to_grid
        if info.inline_range:
            self.config.vel_inline_range = info.inline_range
        if info.xline_range:
            self.config.vel_xline_range = info.xline_range
        if info.cdp_range:
            self.config.vel_cdp_range = info.cdp_range

        # Interpolate velocity to output grid if requested
        if self.config.interpolate_velocity and self._needs_velocity_interpolation():
            self._interpolate_velocity_to_grid()

    def _needs_velocity_interpolation(self) -> bool:
        """Check if velocity model needs interpolation to output grid."""
        from models.velocity_model import VelocityType

        # If velocity is constant or 1D, no spatial interpolation needed
        if self._velocity_model.velocity_type in [VelocityType.CONSTANT, VelocityType.V_OF_Z]:
            return False

        # If velocity is spatially varying (2D/3D), may need interpolation
        return True

    def _interpolate_velocity_to_grid(self):
        """Interpolate 2D/3D velocity model to the configured output grid.

        Uses velocity spatial info (inline_range, xline_range, cdp_range) from
        the config to properly map velocity locations to output grid locations.
        """
        from scipy.interpolate import interp1d
        from models.velocity_model import VelocityType, create_2d_velocity

        if self._velocity_model.velocity_type != VelocityType.V_OF_XZ:
            return  # Only 2D models need interpolation

        # Get target inline positions from selected inlines
        target_inlines = np.array(self.config.inline_numbers, dtype=np.float32)

        # Source velocity x-axis (could be CDP, inline, etc.)
        src_x = self._velocity_model.x_axis
        src_t = self._velocity_model.z_axis
        src_v = self._velocity_model.data  # shape: (n_time, n_x)

        if len(target_inlines) == 0:
            logger.warning("No target inlines for velocity interpolation")
            return

        # Log velocity spatial info for debugging
        vel_inline_range = getattr(self.config, 'vel_inline_range', None)
        vel_xline_range = getattr(self.config, 'vel_xline_range', None)
        vel_cdp_range = getattr(self.config, 'vel_cdp_range', None)

        logger.info(
            f"Velocity spatial info - Inline: {vel_inline_range}, "
            f"Xline: {vel_xline_range}, CDP: {vel_cdp_range}"
        )
        logger.info(
            f"Output grid - Inline: {self.config.inline_min}-{self.config.inline_max}, "
            f"Xline: {self.config.xline_min}-{self.config.xline_max}"
        )

        # Determine the mapping between velocity x-axis and target inlines
        # Priority: use inline_range from velocity if available, else CDP
        effective_src_x = src_x
        if vel_inline_range is not None:
            # Velocity has inline info - use it directly for interpolation
            logger.info(f"Using velocity inline range {vel_inline_range} for interpolation")
        elif vel_cdp_range is not None and self.config.inline_min is not None:
            # Map CDP range to inline range if both are available
            # This assumes a linear relationship between CDP and inline
            data_inline_range = (self.config.inline_min, self.config.inline_max)
            cdp_min, cdp_max = vel_cdp_range
            inline_min, inline_max = data_inline_range

            if cdp_max > cdp_min and inline_max > inline_min:
                # Linear mapping from CDP to inline
                scale = (inline_max - inline_min) / (cdp_max - cdp_min)
                effective_src_x = inline_min + (src_x - cdp_min) * scale
                logger.info(
                    f"Mapped velocity CDP range {vel_cdp_range} to inline range "
                    f"({effective_src_x.min():.0f}, {effective_src_x.max():.0f})"
                )

        # Interpolate velocity at each target inline
        n_time = len(src_t)
        n_target = len(target_inlines)
        interp_v = np.zeros((n_time, n_target), dtype=np.float32)

        for t_idx in range(n_time):
            # Interpolate velocity at this time along x-axis
            f = interp1d(effective_src_x, src_v[t_idx, :], kind='linear',
                         bounds_error=False, fill_value='extrapolate')
            interp_v[t_idx, :] = f(target_inlines)

        # Create new interpolated velocity model
        self._velocity_model = create_2d_velocity(
            data=interp_v,
            z_axis=src_t.copy(),
            x_axis=target_inlines,
            is_time=True,
            metadata={
                **self._velocity_model.metadata,
                'interpolated_to_grid': True,
                'original_x_range': (float(src_x.min()), float(src_x.max())),
                'vel_inline_range': vel_inline_range,
                'vel_xline_range': vel_xline_range,
                'vel_cdp_range': vel_cdp_range,
                'target_inlines': target_inlines.tolist(),
            }
        )

        logger.info(f"Interpolated velocity to {n_target} target inlines")

    def _identify_gathers(self) -> Dict[int, List[Dict[str, Any]]]:
        """
        Identify CDP gathers contributing to selected inlines.

        Uses the configured inline_header and xline_header field names.
        Falls back to using headers.parquet if ensemble_index doesn't have inline column.

        Returns:
            Dictionary mapping inline_no -> list of gather info dicts
        """
        dataset_path = Path(self.config.dataset_path)
        ensemble_path = dataset_path / "ensemble_index.parquet"
        headers_path = dataset_path / "headers.parquet"

        if not ensemble_path.exists():
            raise FileNotFoundError(f"Ensemble index not found: {ensemble_path}")

        # Load ensemble index
        ensemble_df = pd.read_parquet(ensemble_path)
        logger.info(f"Ensemble index columns: {list(ensemble_df.columns)}")

        # Use configured inline header, with fallbacks
        configured_inline = getattr(self.config, 'inline_header', 'INLINE_NO')
        configured_xline = getattr(self.config, 'xline_header', 'XLINE_NO')

        # Check if inline column exists in ensemble_index
        inline_col = None
        if configured_inline in ensemble_df.columns:
            inline_col = configured_inline
        else:
            for col in ['INLINE_NO', 'inline_no', 'INLINE_3D', 'Inline', 'INLINE', 'FieldRecord', 'FLDR']:
                if col in ensemble_df.columns:
                    inline_col = col
                    break

        # If inline not in ensemble_index, we need to use headers.parquet
        if inline_col is None:
            logger.info(
                f"Inline column '{configured_inline}' not in ensemble_index. "
                f"Loading from headers.parquet..."
            )
            return self._identify_gathers_from_headers(
                dataset_path, ensemble_df, configured_inline, configured_xline
            )

        # Get xline column for gather identification (optional)
        xline_col = None
        if configured_xline in ensemble_df.columns:
            xline_col = configured_xline
        else:
            for col in ['XLINE_NO', 'CROSSLINE_3D', 'XL', 'crossline', 'CDP', 'cdp']:
                if col in ensemble_df.columns:
                    xline_col = col
                    break

        logger.info(f"Using inline column: {inline_col}, xline column: {xline_col}")

        # Build gather info by inline
        gather_info = {}
        for inline in self.config.inline_numbers:
            mask = ensemble_df[inline_col] == inline
            inline_df = ensemble_df[mask]

            gathers = []
            for _, row in inline_df.iterrows():
                gather_dict = {
                    'ensemble_id': row.get('ensemble_id', row.name),
                    'start_trace': row['start_trace'],
                    'end_trace': row['end_trace'],
                    'n_traces': row.get('n_traces', row['end_trace'] - row['start_trace']),
                    'cdp': row.get('CDP', row.get('cdp', row.get('ensemble_value', row.get('ensemble_id', 0)))),
                    'inline': inline,
                }
                # Add xline if available
                if xline_col and xline_col in row.index:
                    gather_dict['xline'] = row[xline_col]
                gathers.append(gather_dict)
            gather_info[inline] = gathers

        total_gathers = sum(len(g) for g in gather_info.values())
        logger.info(f"Identified {total_gathers} gathers for {len(self.config.inline_numbers)} inlines")

        return gather_info

    def _identify_gathers_from_headers(
        self,
        dataset_path: Path,
        ensemble_df: pd.DataFrame,
        inline_header: str,
        xline_header: str,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Identify gathers using headers.parquet when ensemble_index doesn't have inline column.

        This handles cases where data was sorted by CDP/FieldRecord but we want to
        select by inline number.
        """
        headers_path = dataset_path / "headers.parquet"

        if not headers_path.exists():
            raise FileNotFoundError(
                f"Headers file not found: {headers_path}. "
                f"Cannot identify gathers without inline column in ensemble_index."
            )

        # Load headers - only columns we need
        logger.info(f"Loading headers.parquet to find inline '{inline_header}'...")
        headers_df = pd.read_parquet(headers_path)

        # Find inline column in headers
        inline_col = None
        if inline_header in headers_df.columns:
            inline_col = inline_header
        else:
            for col in ['INLINE_NO', 'inline_no', 'INLINE_3D', 'Inline', 'INLINE', 'FieldRecord', 'FLDR']:
                if col in headers_df.columns:
                    inline_col = col
                    break

        if inline_col is None:
            raise ValueError(
                f"Inline column '{inline_header}' not found in headers. "
                f"Available columns: {list(headers_df.columns)}"
            )

        # Find xline column
        xline_col = None
        if xline_header in headers_df.columns:
            xline_col = xline_header
        else:
            for col in ['XLINE_NO', 'CROSSLINE_3D', 'XL', 'crossline', 'CDP', 'cdp']:
                if col in headers_df.columns:
                    xline_col = col
                    break

        logger.info(f"Found inline column: {inline_col}, xline column: {xline_col}")

        # Build gather info by finding trace ranges for each inline
        gather_info = {}

        for inline in self.config.inline_numbers:
            # Find all traces with this inline number
            mask = headers_df[inline_col] == inline
            inline_traces = headers_df[mask]

            if len(inline_traces) == 0:
                logger.warning(f"No traces found for inline {inline}")
                gather_info[inline] = []
                continue

            # Group by xline/CDP to form gathers
            if xline_col:
                grouped = inline_traces.groupby(xline_col)
            else:
                # No xline - treat all traces as one gather
                grouped = [(0, inline_traces)]

            gathers = []
            for xline_val, group in grouped:
                # Get trace indices (position in the full dataset)
                trace_indices = group.index.tolist()
                if len(trace_indices) == 0:
                    continue

                gather_dict = {
                    'ensemble_id': f"{inline}_{xline_val}",
                    'start_trace': min(trace_indices),
                    'end_trace': max(trace_indices) + 1,
                    'n_traces': len(trace_indices),
                    'cdp': int(xline_val) if xline_col else 0,
                    'inline': inline,
                    'xline': int(xline_val) if xline_col else 0,
                    'trace_indices': trace_indices,  # Keep for non-contiguous access
                }
                gathers.append(gather_dict)

            gather_info[inline] = gathers

        total_gathers = sum(len(g) for g in gather_info.values())
        logger.info(
            f"Identified {total_gathers} gathers for {len(self.config.inline_numbers)} inlines "
            f"(from headers.parquet)"
        )

        return gather_info

    def _stack_inlines(
        self,
        gather_info: Dict[int, List[Dict[str, Any]]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, Tuple[int, int]]]:
        """
        Stack all selected inlines.

        Returns:
            Tuple of (stacked_data, fold_data, cdp_axis, inline_ranges)
            stacked_data shape: (n_samples, n_total_cdps)
            inline_ranges: dict mapping inline_num -> (start_idx, end_idx)
        """
        dataset_path = Path(self.config.dataset_path)

        # Open data stores
        traces_zarr = zarr.open(dataset_path / "traces.zarr", mode='r')
        headers_df = pd.read_parquet(dataset_path / "headers.parquet")

        n_samples = traces_zarr.shape[0]
        sample_interval_ms = self._get_sample_interval(dataset_path)

        # Setup NMO and stacker
        nmo_config = NMOConfig(
            stretch_mute_factor=self.config.stretch_mute,
            velocity_type=self.config.velocity_type,
        )
        nmo_processor = NMOProcessor(nmo_config, self._velocity_model)

        stack_config = StackConfig(
            method=self.config.stack_method,
            min_fold=self.config.min_fold,
        )
        stacker = CDPStacker(stack_config, nmo_processor=nmo_processor)

        # Count total CDPs
        total_cdps = sum(len(gathers) for gathers in gather_info.values())

        # Allocate output arrays
        all_stacked = []
        all_fold = []
        all_cdps = []
        inline_ranges = {}  # Track where each inline's data starts/ends

        cdp_idx = 0
        for inline in self.config.inline_numbers:
            if self._cancelled:
                break

            gathers = gather_info.get(inline, [])
            inline_start_idx = cdp_idx  # Track start of this inline

            for gather in gathers:
                if self._cancelled:
                    break

                self._report_progress(
                    'stacking',
                    cdp_idx,
                    total_cdps,
                    f"Inline {inline}, CDP {gather['cdp']}",
                    inline=inline,
                    cdp=gather['cdp']
                )

                # Load gather traces - handle both contiguous and non-contiguous cases
                if 'trace_indices' in gather:
                    # Non-contiguous traces (from headers.parquet lookup)
                    trace_indices = gather['trace_indices']
                    traces = np.array(traces_zarr[:, trace_indices])
                    gather_headers = headers_df.iloc[trace_indices]
                    n_traces = len(trace_indices)
                else:
                    # Contiguous traces (from ensemble_index)
                    start = gather['start_trace']
                    end = gather['end_trace']
                    traces = np.array(traces_zarr[:, start:end])
                    gather_headers = headers_df.iloc[start:end]
                    n_traces = end - start

                # Get offsets
                offset_col = None
                for col in ['OFFSET', 'offset', 'Offset']:
                    if col in gather_headers.columns:
                        offset_col = col
                        break

                if offset_col:
                    offsets = gather_headers[offset_col].values.astype(np.float32)
                else:
                    # Fallback: create dummy offsets
                    offsets = np.arange(n_traces, dtype=np.float32) * 100

                # Stack
                result = stacker.stack_gather_with_nmo(
                    traces,
                    offsets,
                    sample_interval_ms,
                    cdp=gather['cdp']
                )

                all_stacked.append(result.trace)
                all_fold.append(result.fold)
                all_cdps.append(gather['cdp'])

                cdp_idx += 1

            # Record inline range (even if empty)
            inline_ranges[inline] = (inline_start_idx, cdp_idx)

        # Combine results
        if all_stacked:
            stacked_data = np.column_stack(all_stacked)
            fold_data = np.column_stack(all_fold)
            cdp_axis = np.array(all_cdps)
        else:
            stacked_data = np.zeros((n_samples, 0), dtype=np.float32)
            fold_data = np.zeros((n_samples, 0), dtype=np.int32)
            cdp_axis = np.array([], dtype=np.int32)

        return stacked_data, fold_data, cdp_axis, inline_ranges

    def _get_sample_interval(self, dataset_path: Path) -> float:
        """Get sample interval in milliseconds from metadata."""
        metadata_path = dataset_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            # Sample interval stored in seconds, convert to ms
            dt = metadata.get('sample_interval', 0.004)
            return dt * 1000 if dt < 1 else dt  # Handle if already in ms
        return 4.0  # Default 4ms

    def _write_output(
        self,
        stacked_data: np.ndarray,
        fold_data: np.ndarray,
        cdp_axis: np.ndarray,
        inline_ranges: Dict[int, Tuple[int, int]]
    ) -> str:
        """
        Write stacked data to output files.

        Args:
            stacked_data: Stacked trace data (n_samples, n_cdps)
            fold_data: Fold data (n_samples, n_cdps)
            cdp_axis: CDP numbers for each trace
            inline_ranges: Dict mapping inline_num -> (start_idx, end_idx)

        Returns:
            Output path
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_name = self.config.output_name
        output_zarr_path = output_dir / f"{output_name}.zarr"
        output_meta_path = output_dir / f"{output_name}_metadata.json"

        # Write Zarr
        zarr_store = zarr.open(
            str(output_zarr_path),
            mode='w',
            shape=stacked_data.shape,
            dtype=np.float32,
            chunks=(stacked_data.shape[0], min(64, stacked_data.shape[1]))
        )
        zarr_store[:] = stacked_data

        # Convert inline_ranges to JSON-serializable format
        # Keys must be strings in JSON
        inline_ranges_json = {str(k): list(v) for k, v in inline_ranges.items()}

        # Write metadata
        metadata = {
            'type': 'qc_stack',
            'source_dataset': str(self.config.dataset_path),
            'inline_numbers': self.config.inline_numbers,
            'inline_ranges': inline_ranges_json,  # Store inline boundaries
            'velocity_file': self.config.velocity_file,
            'velocity_type': self.config.velocity_type,
            'stretch_mute': self.config.stretch_mute,
            'stack_method': self.config.stack_method,
            'min_fold': self.config.min_fold,
            'n_samples': stacked_data.shape[0],
            'n_cdps': stacked_data.shape[1],
            'cdp_numbers': cdp_axis.tolist(),
            'sample_interval': self._get_sample_interval(Path(self.config.dataset_path)) / 1000,
        }

        with open(output_meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Write fold as separate array
        fold_path = output_dir / f"{output_name}_fold.zarr"
        fold_store = zarr.open(str(fold_path), mode='w', shape=fold_data.shape, dtype=np.int32)
        fold_store[:] = fold_data

        logger.info(f"Wrote QC stack to {output_zarr_path}")
        logger.info(f"Inline ranges: {inline_ranges}")
        return str(output_zarr_path)


# =============================================================================
# Worker Thread for Background Execution
# =============================================================================

from PyQt6.QtCore import QThread


class QCStackingWorker(QThread):
    """
    Worker thread for running QC stacking in background.

    Usage:
        worker = QCStackingWorker(config)
        worker.progress_updated.connect(on_progress)
        worker.finished_with_result.connect(on_complete)
        worker.start()
    """

    progress_updated = pyqtSignal(object)  # StackingProgress
    finished_with_result = pyqtSignal(str)  # Output path
    error_occurred = pyqtSignal(str)

    def __init__(self, config: 'QCStackingConfig', parent=None):
        super().__init__(parent)
        self.config = config
        self._engine: Optional[QCStackingEngine] = None

    def run(self):
        """Execute stacking in thread."""
        self._engine = QCStackingEngine(self.config)
        self._engine.progress_updated.connect(self.progress_updated.emit)

        try:
            result = self._engine.run()
            if result:
                self.finished_with_result.emit(result)
            else:
                self.error_occurred.emit("Stacking cancelled or failed")
        except Exception as e:
            logger.exception("Stacking worker error")
            self.error_occurred.emit(str(e))

    def cancel(self):
        """Request cancellation."""
        if self._engine:
            self._engine.cancel()
