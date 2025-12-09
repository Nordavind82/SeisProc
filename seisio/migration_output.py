"""
Migration Output Manager

Handles generation of multiple output volumes for bin-by-bin migration:
- One output volume per bin (e.g., migrated_near_offset.sgy)
- Full stack output (sum of all bins)
- Fold volume (trace count per output sample)
- Output header population (INLINE, XLINE, OFFSET, AZIMUTH, CDP_X, CDP_Y)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path
import numpy as np
import logging

from models.migration_config import OutputGrid
from models.binning import BinningTable, OffsetAzimuthBin

logger = logging.getLogger(__name__)


@dataclass
class OutputVolumeInfo:
    """Information about an output volume."""
    name: str
    filepath: str
    bin_name: Optional[str] = None  # None for stack/fold volumes
    offset_center: Optional[float] = None
    azimuth_center: Optional[float] = None
    n_traces_migrated: int = 0
    is_complete: bool = False


@dataclass
class MigrationOutputManager:
    """
    Manages multiple output volumes for migration.

    Creates and manages:
    - Per-bin migrated volumes
    - Full stack volume (optional)
    - Fold volume (optional)
    """
    output_directory: str
    output_grid: OutputGrid
    binning_table: BinningTable
    base_name: str = "migrated"
    output_format: str = "npy"  # 'npy', 'segy', 'zarr'
    create_stack: bool = True
    create_fold: bool = False
    compress: bool = False

    # Internal state
    _volumes: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    _fold_volumes: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    _stack_volume: Optional[np.ndarray] = field(default=None, repr=False)
    _stack_fold: Optional[np.ndarray] = field(default=None, repr=False)
    _volume_info: Dict[str, OutputVolumeInfo] = field(default_factory=dict)
    _initialized: bool = False

    def __post_init__(self):
        """Create output directory if needed."""
        Path(self.output_directory).mkdir(parents=True, exist_ok=True)

    def initialize(self):
        """
        Initialize output volumes.

        Creates empty arrays for all output volumes.
        """
        if self._initialized:
            logger.warning("Output manager already initialized")
            return

        shape = (
            self.output_grid.n_time,
            self.output_grid.n_inline,
            self.output_grid.n_xline,
        )

        logger.info(f"Initializing output volumes with shape {shape}")

        # Create volume for each enabled bin
        for bin_obj in self.binning_table.bins:
            if not bin_obj.enabled:
                continue

            self._volumes[bin_obj.name] = np.zeros(shape, dtype=np.float32)

            if self.create_fold:
                self._fold_volumes[bin_obj.name] = np.zeros(shape, dtype=np.int32)

            # Record volume info
            self._volume_info[bin_obj.name] = OutputVolumeInfo(
                name=bin_obj.name,
                filepath=self._get_volume_path(bin_obj.name),
                bin_name=bin_obj.name,
                offset_center=(bin_obj.offset_min + bin_obj.offset_max) / 2,
                azimuth_center=(bin_obj.azimuth_min + bin_obj.azimuth_max) / 2,
            )

        # Create stack volume if requested
        if self.create_stack:
            self._stack_volume = np.zeros(shape, dtype=np.float32)
            self._stack_fold = np.zeros(shape, dtype=np.int32)

            self._volume_info['stack'] = OutputVolumeInfo(
                name='stack',
                filepath=self._get_volume_path('stack'),
            )

        self._initialized = True
        logger.info(f"Initialized {len(self._volumes)} output volumes")

    def _get_volume_path(self, name: str) -> str:
        """Get filepath for a volume."""
        ext = {
            'npy': '.npy',
            'segy': '.sgy',
            'zarr': '.zarr',
        }.get(self.output_format, '.npy')

        filename = f"{self.base_name}_{name}{ext}"
        return str(Path(self.output_directory) / filename)

    def add_migrated_data(
        self,
        bin_name: str,
        data: np.ndarray,
        fold: Optional[np.ndarray] = None,
    ):
        """
        Add migrated data for a bin.

        Args:
            bin_name: Name of the bin
            data: Migrated data array (n_time, n_inline, n_xline)
            fold: Optional fold array
        """
        if not self._initialized:
            raise RuntimeError("Output manager not initialized")

        if bin_name not in self._volumes:
            raise ValueError(f"Unknown bin: {bin_name}")

        # Add to bin volume
        self._volumes[bin_name] += data

        # Update fold
        if self.create_fold and fold is not None:
            self._fold_volumes[bin_name] += fold

        # Add to stack
        if self.create_stack:
            self._stack_volume += data
            if fold is not None:
                self._stack_fold += fold

        # Update info
        info = self._volume_info[bin_name]
        info.n_traces_migrated += 1

        logger.debug(f"Added data to bin {bin_name}")

    def accumulate_to_bin(
        self,
        bin_name: str,
        time_idx: int,
        inline_idx: int,
        xline_idx: int,
        value: float,
    ):
        """
        Accumulate a single sample value to a bin volume.

        Used for fine-grained accumulation during migration.

        Args:
            bin_name: Bin name
            time_idx: Time sample index
            inline_idx: Inline index
            xline_idx: Crossline index
            value: Value to accumulate
        """
        if not self._initialized:
            raise RuntimeError("Output manager not initialized")

        if bin_name in self._volumes:
            self._volumes[bin_name][time_idx, inline_idx, xline_idx] += value

            if self.create_fold:
                self._fold_volumes[bin_name][time_idx, inline_idx, xline_idx] += 1

            if self.create_stack:
                self._stack_volume[time_idx, inline_idx, xline_idx] += value
                self._stack_fold[time_idx, inline_idx, xline_idx] += 1

    def get_volume(self, bin_name: str) -> np.ndarray:
        """Get volume data for a bin."""
        if bin_name == 'stack':
            return self._stack_volume
        return self._volumes.get(bin_name)

    def get_fold(self, bin_name: str) -> Optional[np.ndarray]:
        """Get fold volume for a bin."""
        if bin_name == 'stack':
            return self._stack_fold
        return self._fold_volumes.get(bin_name)

    def normalize_by_fold(self, min_fold: int = 1):
        """
        Normalize all volumes by fold count.

        Args:
            min_fold: Minimum fold threshold (set to zero if below)
        """
        logger.info(f"Normalizing volumes by fold (min_fold={min_fold})")

        for bin_name, volume in self._volumes.items():
            fold = self._fold_volumes.get(bin_name)
            if fold is not None:
                mask = fold >= min_fold
                volume[mask] /= fold[mask]
                volume[~mask] = 0.0

        if self.create_stack and self._stack_fold is not None:
            mask = self._stack_fold >= min_fold
            self._stack_volume[mask] /= self._stack_fold[mask]
            self._stack_volume[~mask] = 0.0

    def finalize_bin(self, bin_name: str):
        """
        Finalize a bin volume (mark as complete).

        Args:
            bin_name: Bin name
        """
        if bin_name in self._volume_info:
            self._volume_info[bin_name].is_complete = True
            logger.info(f"Finalized bin {bin_name}")

    def save_all(self):
        """Save all output volumes to files."""
        logger.info(f"Saving {len(self._volumes)} volumes to {self.output_directory}")

        for bin_name, volume in self._volumes.items():
            filepath = self._volume_info[bin_name].filepath
            self._save_volume(volume, filepath, bin_name)

            # Save fold volume if requested
            if self.create_fold and bin_name in self._fold_volumes:
                fold_path = filepath.replace('.', '_fold.')
                self._save_volume(self._fold_volumes[bin_name], fold_path, f"{bin_name}_fold")

        # Save stack volume
        if self.create_stack and self._stack_volume is not None:
            filepath = self._volume_info['stack'].filepath
            self._save_volume(self._stack_volume, filepath, 'stack')

            if self.create_fold:
                fold_path = filepath.replace('.', '_fold.')
                self._save_volume(self._stack_fold, fold_path, 'stack_fold')

        logger.info("All volumes saved")

    def _save_volume(self, volume: np.ndarray, filepath: str, name: str):
        """Save a single volume."""
        if self.output_format == 'npy':
            np.save(filepath, volume)
            logger.debug(f"Saved {name} to {filepath}")

        elif self.output_format == 'zarr':
            import zarr
            zarr.save(filepath, volume)
            logger.debug(f"Saved {name} to {filepath}")

        elif self.output_format == 'segy':
            # SEG-Y output requires more complex handling
            # For now, fall back to numpy
            logger.warning("SEG-Y output not yet implemented, saving as numpy")
            np_path = filepath.replace('.sgy', '.npy')
            np.save(np_path, volume)

        else:
            raise ValueError(f"Unknown output format: {self.output_format}")

    def save_bin(self, bin_name: str):
        """Save a single bin volume."""
        if bin_name not in self._volumes:
            raise ValueError(f"Unknown bin: {bin_name}")

        volume = self._volumes[bin_name]
        filepath = self._volume_info[bin_name].filepath
        self._save_volume(volume, filepath, bin_name)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of output volumes."""
        return {
            'output_directory': self.output_directory,
            'n_bins': len(self._volumes),
            'output_format': self.output_format,
            'create_stack': self.create_stack,
            'create_fold': self.create_fold,
            'volumes': {
                name: {
                    'filepath': info.filepath,
                    'bin_name': info.bin_name,
                    'offset_center': info.offset_center,
                    'azimuth_center': info.azimuth_center,
                    'n_traces_migrated': info.n_traces_migrated,
                    'is_complete': info.is_complete,
                }
                for name, info in self._volume_info.items()
            }
        }

    def get_output_headers(
        self,
        bin_name: str,
        inline_idx: int,
        xline_idx: int,
    ) -> Dict[str, Any]:
        """
        Get output trace headers for a position.

        Args:
            bin_name: Bin name
            inline_idx: Inline index
            xline_idx: Crossline index

        Returns:
            Dictionary of header values
        """
        # Get inline/xline numbers
        inline = self.output_grid.inline_start + inline_idx
        xline = self.output_grid.xline_start + xline_idx

        # Get world coordinates
        cdp_x, cdp_y = self.output_grid.get_coordinates(inline_idx, xline_idx)

        headers = {
            'INLINE_3D': inline,
            'CROSSLINE_3D': xline,
            'CDP_X': cdp_x,
            'CDP_Y': cdp_y,
        }

        # Add bin information if available
        if bin_name in self._volume_info:
            info = self._volume_info[bin_name]
            if info.offset_center is not None:
                headers['OFFSET'] = info.offset_center
            if info.azimuth_center is not None:
                headers['AZIMUTH'] = info.azimuth_center

        return headers


def create_output_manager(
    job_config: 'MigrationJobConfig',
) -> MigrationOutputManager:
    """
    Create output manager from job configuration.

    Args:
        job_config: Migration job configuration

    Returns:
        Configured MigrationOutputManager
    """
    return MigrationOutputManager(
        output_directory=job_config.output_directory,
        output_grid=job_config.get_output_grid(),
        binning_table=job_config.get_binning_table(),
        base_name=job_config.name.replace(' ', '_').lower(),
        output_format=job_config.output_format,
        create_stack=job_config.create_stack_volume,
        create_fold=job_config.create_fold_volume,
        compress=job_config.compress_output,
    )
