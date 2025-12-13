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
            stacked_data, fold_data, cdp_axis = self._stack_inlines(gather_info)

            if self._cancelled:
                return None

            # Phase 4: Write output
            self._report_progress('writing', 3, 3, "Writing output...")
            output_path = self._write_output(stacked_data, fold_data, cdp_axis)

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
        """Load velocity model from configured file."""
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

        # Filter to selected inlines
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
                })
            gather_info[inline] = gathers

        total_gathers = sum(len(g) for g in gather_info.values())
        logger.info(f"Identified {total_gathers} gathers for {len(self.config.inline_numbers)} inlines")

        return gather_info

    def _stack_inlines(
        self,
        gather_info: Dict[int, List[Dict[str, Any]]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Stack all selected inlines.

        Returns:
            Tuple of (stacked_data, fold_data, cdp_axis)
            stacked_data shape: (n_samples, n_total_cdps)
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

        cdp_idx = 0
        for inline in self.config.inline_numbers:
            if self._cancelled:
                break

            gathers = gather_info.get(inline, [])

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

                # Load gather traces
                start = gather['start_trace']
                end = gather['end_trace']
                traces = np.array(traces_zarr[:, start:end])

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
                    # Fallback: create dummy offsets
                    offsets = np.arange(end - start, dtype=np.float32) * 100

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

        # Combine results
        if all_stacked:
            stacked_data = np.column_stack(all_stacked)
            fold_data = np.column_stack(all_fold)
            cdp_axis = np.array(all_cdps)
        else:
            stacked_data = np.zeros((n_samples, 0), dtype=np.float32)
            fold_data = np.zeros((n_samples, 0), dtype=np.int32)
            cdp_axis = np.array([], dtype=np.int32)

        return stacked_data, fold_data, cdp_axis

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
        cdp_axis: np.ndarray
    ) -> str:
        """
        Write stacked data to output files.

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

        # Write metadata
        metadata = {
            'type': 'qc_stack',
            'source_dataset': str(self.config.dataset_path),
            'inline_numbers': self.config.inline_numbers,
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
