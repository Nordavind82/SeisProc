"""
Base Migrator - Abstract Interface for Migration Algorithms

Defines the standard interface that all migration implementations must follow.
Supports both GPU and CPU implementations with a consistent API.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable, List
import numpy as np
import logging

from models.velocity_model import VelocityModel
from models.migration_config import MigrationConfig
from models.migration_geometry import MigrationGeometry
from models.seismic_data import SeismicData

logger = logging.getLogger(__name__)

# Type alias for progress callbacks: (current, total, message) -> None
ProgressCallback = Callable[[int, int, str], None]


class BaseMigrator(ABC):
    """
    Abstract base class for all migration algorithms.

    Defines the interface for:
    - Single gather migration
    - Multi-gather (dataset) migration
    - Memory estimation
    - Progress reporting

    All migrators must:
    1. Be immutable (don't modify input data)
    2. Return new result objects
    3. Validate parameters in __init__
    4. Support progress callbacks
    5. Provide memory estimation
    """

    def __init__(
        self,
        velocity_model: VelocityModel,
        config: MigrationConfig,
        **kwargs,
    ):
        """
        Initialize migrator with velocity model and configuration.

        Args:
            velocity_model: Velocity model for traveltime computation
            config: Migration configuration parameters
            **kwargs: Additional implementation-specific parameters
        """
        self.velocity_model = velocity_model
        self.config = config
        self.params = kwargs
        self._progress_callback: Optional[ProgressCallback] = None

        # Validate inputs
        self._validate_inputs()

        logger.info(f"Initialized {self.__class__.__name__}")

    @abstractmethod
    def _validate_inputs(self):
        """
        Validate velocity model and configuration.

        Raises:
            ValueError: If inputs are invalid
        """
        pass

    def set_progress_callback(
        self,
        callback: Optional[ProgressCallback]
    ) -> 'BaseMigrator':
        """
        Set progress callback for processing status updates.

        Args:
            callback: Function(current, total, message) to receive progress updates.
                     Set to None to disable progress reporting.

        Returns:
            self for method chaining
        """
        self._progress_callback = callback
        return self

    def _report_progress(self, current: int, total: int, message: str = ""):
        """
        Report progress to callback if set.

        Args:
            current: Current progress value
            total: Total items to process
            message: Optional status message
        """
        if self._progress_callback is not None:
            self._progress_callback(current, total, message)

    @abstractmethod
    def migrate_gather(
        self,
        gather: SeismicData,
        geometry: MigrationGeometry,
    ) -> np.ndarray:
        """
        Migrate a single shot/receiver gather.

        Args:
            gather: Pre-stack seismic gather (n_samples, n_traces)
            geometry: Source and receiver coordinates for traces

        Returns:
            Partial image contribution as numpy array
            Shape depends on output grid configuration
        """
        pass

    @abstractmethod
    def migrate_dataset(
        self,
        gathers: List[SeismicData],
        geometries: List[MigrationGeometry],
        batch_size: int = 1,
    ) -> 'MigrationResult':
        """
        Migrate entire pre-stack dataset.

        Args:
            gathers: List of shot/receiver gathers
            geometries: List of geometry objects (one per gather)
            batch_size: Number of gathers to process per batch

        Returns:
            Complete MigrationResult with image and metadata
        """
        pass

    @abstractmethod
    def estimate_memory_gb(
        self,
        n_traces: int,
        n_samples: int,
    ) -> float:
        """
        Estimate GPU/memory requirement for migration.

        Args:
            n_traces: Number of input traces
            n_samples: Samples per trace

        Returns:
            Estimated memory requirement in GB
        """
        pass

    def can_process(
        self,
        n_traces: int,
        n_samples: int,
        available_memory_gb: float,
    ) -> bool:
        """
        Check if data can be processed with available memory.

        Args:
            n_traces: Number of input traces
            n_samples: Samples per trace
            available_memory_gb: Available GPU/system memory

        Returns:
            True if processing is feasible
        """
        required = self.estimate_memory_gb(n_traces, n_samples)
        # Use 80% safety margin
        return required < (available_memory_gb * 0.8)

    def calculate_batch_size(
        self,
        n_traces_per_gather: int,
        n_samples: int,
        available_memory_gb: float,
    ) -> int:
        """
        Calculate optimal batch size for available memory.

        Args:
            n_traces_per_gather: Traces per gather
            n_samples: Samples per trace
            available_memory_gb: Available memory

        Returns:
            Recommended number of gathers per batch
        """
        memory_per_gather = self.estimate_memory_gb(n_traces_per_gather, n_samples)
        if memory_per_gather <= 0:
            return 1

        # Use 70% of available memory for safety
        usable_memory = available_memory_gb * 0.7
        batch_size = max(1, int(usable_memory / memory_per_gather))

        return batch_size

    @abstractmethod
    def get_description(self) -> str:
        """Get human-readable description of migrator and parameters."""
        pass

    def get_status(self) -> Dict[str, Any]:
        """
        Get migrator status for UI display.

        Returns:
            Dictionary with status information
        """
        return {
            'class': self.__class__.__name__,
            'velocity_type': str(self.velocity_model.velocity_type),
            'output_grid': f"{self.config.output_grid.n_inline}x{self.config.output_grid.n_xline}x{self.config.output_grid.n_time}",
            'max_aperture': self.config.max_aperture_m,
            'max_angle': self.config.max_angle_deg,
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize migrator configuration for reconstruction.

        Returns:
            Dictionary with class info and parameters
        """
        return {
            'class_name': self.__class__.__name__,
            'module': self.__class__.__module__,
            'velocity_model': self.velocity_model.to_dict(),
            'config': self.config.to_dict(),
            'params': self.params,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"velocity={self.velocity_model}, "
            f"aperture={self.config.max_aperture_m}m)"
        )


class MigrationResult:
    """
    Container for migration output.

    Attributes:
        image: Migrated image array (n_time, n_inline, n_xline)
        fold: Fold count per output sample
        config: Configuration used for migration
        metadata: Processing metadata and statistics
    """

    def __init__(
        self,
        image: np.ndarray,
        fold: Optional[np.ndarray] = None,
        config: Optional[MigrationConfig] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize migration result.

        Args:
            image: Migrated image array
            fold: Optional fold count array
            config: Migration configuration used
            metadata: Additional metadata
        """
        self.image = image.astype(np.float32)
        self.fold = fold.astype(np.int32) if fold is not None else None
        self.config = config
        self.metadata = metadata or {}

    @property
    def shape(self):
        """Image shape."""
        return self.image.shape

    @property
    def n_time(self) -> int:
        """Number of time samples."""
        return self.image.shape[0]

    @property
    def n_inline(self) -> int:
        """Number of inlines."""
        return self.image.shape[1]

    @property
    def n_xline(self) -> int:
        """Number of crosslines."""
        return self.image.shape[2]

    def get_inline_section(self, xline_idx: int) -> np.ndarray:
        """Extract inline section at given crossline index."""
        return self.image[:, :, xline_idx]

    def get_xline_section(self, inline_idx: int) -> np.ndarray:
        """Extract crossline section at given inline index."""
        return self.image[:, inline_idx, :]

    def get_time_slice(self, time_idx: int) -> np.ndarray:
        """Extract time slice at given time index."""
        return self.image[time_idx, :, :]

    def apply_fold_normalization(
        self,
        min_fold: int = 1,
    ) -> 'MigrationResult':
        """
        Apply fold normalization to image.

        Args:
            min_fold: Minimum fold threshold (mute below this)

        Returns:
            New MigrationResult with normalized image
        """
        if self.fold is None:
            return self

        # Avoid division by zero
        fold_safe = np.maximum(self.fold, min_fold)
        normalized_image = self.image / fold_safe

        # Mute low fold areas
        normalized_image[self.fold < min_fold] = 0.0

        return MigrationResult(
            image=normalized_image,
            fold=self.fold.copy(),
            config=self.config,
            metadata={
                **self.metadata,
                'fold_normalized': True,
                'min_fold': min_fold,
            }
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get image statistics."""
        return {
            'shape': self.shape,
            'image_min': float(np.min(self.image)),
            'image_max': float(np.max(self.image)),
            'image_mean': float(np.mean(self.image)),
            'image_std': float(np.std(self.image)),
            'image_rms': float(np.sqrt(np.mean(self.image**2))),
            'fold_min': int(np.min(self.fold)) if self.fold is not None else None,
            'fold_max': int(np.max(self.fold)) if self.fold is not None else None,
            'fold_mean': float(np.mean(self.fold)) if self.fold is not None else None,
        }

    def memory_gb(self) -> float:
        """Memory size in GB."""
        total = self.image.nbytes
        if self.fold is not None:
            total += self.fold.nbytes
        return total / (1024**3)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result metadata (not image data)."""
        return {
            'shape': self.shape,
            'has_fold': self.fold is not None,
            'config': self.config.to_dict() if self.config else None,
            'metadata': self.metadata,
            'statistics': self.get_statistics(),
        }

    def __repr__(self) -> str:
        return (
            f"MigrationResult(shape={self.shape}, "
            f"memory={self.memory_gb():.2f}GB)"
        )
