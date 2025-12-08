"""
Dataset navigator - manages multiple loaded datasets with navigation and switching.

Provides centralized management for multi-dataset workflows, enabling users to
load, switch between, and compare multiple seismic datasets efficiently.
"""
import uuid
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any
from collections import OrderedDict
from dataclasses import dataclass, field
from PyQt6.QtCore import QObject, pyqtSignal

from models.lazy_seismic_data import LazySeismicData

# Set up module logger
logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """
    Metadata container for a loaded dataset.

    Attributes:
        dataset_id: Unique identifier (UUID string)
        name: Human-readable dataset name
        source_path: Original file path (SEG-Y)
        storage_path: Zarr storage directory path
        loaded_at: Timestamp when dataset was loaded
        n_traces: Total number of traces
        n_samples: Number of samples per trace
        n_ensembles: Number of ensembles/gathers
        sample_rate: Sample rate in milliseconds
        metadata: Additional metadata from import
    """
    dataset_id: str
    name: str
    source_path: Path
    storage_path: Path
    loaded_at: datetime
    n_traces: int = 0
    n_samples: int = 0
    n_ensembles: int = 0
    sample_rate: float = 2.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'dataset_id': self.dataset_id,
            'name': self.name,
            'source_path': str(self.source_path),
            'storage_path': str(self.storage_path),
            'loaded_at': self.loaded_at.isoformat(),
            'n_traces': self.n_traces,
            'n_samples': self.n_samples,
            'n_ensembles': self.n_ensembles,
            'sample_rate': self.sample_rate,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetInfo':
        """Create from dictionary."""
        return cls(
            dataset_id=data['dataset_id'],
            name=data['name'],
            source_path=Path(data['source_path']),
            storage_path=Path(data['storage_path']),
            loaded_at=datetime.fromisoformat(data['loaded_at']),
            n_traces=data.get('n_traces', 0),
            n_samples=data.get('n_samples', 0),
            n_ensembles=data.get('n_ensembles', 0),
            sample_rate=data.get('sample_rate', 2.0),
            metadata=data.get('metadata', {}),
        )


class DatasetNavigator(QObject):
    """
    Centralized manager for multiple loaded seismic datasets.

    Manages a registry of loaded datasets, handles active dataset switching,
    and provides metadata access without loading full data into memory.

    Signals:
        dataset_added: Emitted when a new dataset is added (dataset_id, info_dict)
        dataset_removed: Emitted when a dataset is removed (dataset_id)
        active_dataset_changed: Emitted when active dataset changes (dataset_id)
        datasets_cleared: Emitted when all datasets are cleared

    Example:
        >>> navigator = DatasetNavigator()
        >>> dataset_id = navigator.add_dataset(
        ...     source_path=Path('/data/survey.sgy'),
        ...     storage_path=Path('/data/survey_zarr'),
        ...     lazy_data=lazy_seismic_data
        ... )
        >>> navigator.set_active_dataset(dataset_id)
        >>> active = navigator.get_active_dataset()
    """

    # Signals
    dataset_added = pyqtSignal(str, dict)          # dataset_id, info_dict
    dataset_removed = pyqtSignal(str)              # dataset_id
    active_dataset_changed = pyqtSignal(str)       # dataset_id (empty string if none)
    datasets_cleared = pyqtSignal()

    def __init__(self, max_cached_datasets: int = 3):
        """
        Initialize dataset navigator.

        Args:
            max_cached_datasets: Maximum number of datasets to keep loaded in memory.
                                 Oldest accessed datasets are unloaded when limit reached.
        """
        super().__init__()

        # Dataset registry: dataset_id -> DatasetInfo
        self._dataset_info: OrderedDict[str, DatasetInfo] = OrderedDict()

        # Loaded data cache: dataset_id -> LazySeismicData (LRU)
        self._loaded_data: OrderedDict[str, LazySeismicData] = OrderedDict()
        self._max_cached_datasets = max_cached_datasets

        # Active dataset tracking
        self._active_dataset_id: Optional[str] = None

        logger.info(f"DatasetNavigator initialized (max_cached={max_cached_datasets})")

    def add_dataset(
        self,
        source_path: Path,
        storage_path: Path,
        lazy_data: LazySeismicData,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a new dataset to the navigator.

        Args:
            source_path: Original file path (SEG-Y)
            storage_path: Zarr storage directory path
            lazy_data: LazySeismicData instance for the dataset
            name: Optional human-readable name (defaults to filename)
            metadata: Optional additional metadata

        Returns:
            dataset_id: Unique identifier for the dataset

        Emits:
            dataset_added: With dataset_id and info dictionary
        """
        # Generate unique ID
        dataset_id = str(uuid.uuid4())

        # Default name from filename
        if name is None:
            name = source_path.stem if source_path else f"Dataset_{dataset_id[:8]}"

        # Extract info from lazy data
        n_ensembles = lazy_data.get_ensemble_count() if lazy_data else 0

        # Create dataset info
        info = DatasetInfo(
            dataset_id=dataset_id,
            name=name,
            source_path=Path(source_path) if source_path else Path(),
            storage_path=Path(storage_path) if storage_path else Path(),
            loaded_at=datetime.now(),
            n_traces=lazy_data.n_traces if lazy_data else 0,
            n_samples=lazy_data.n_samples if lazy_data else 0,
            n_ensembles=n_ensembles,
            sample_rate=lazy_data.sample_rate if lazy_data else 2.0,
            metadata=metadata or {},
        )

        # Store info and data
        self._dataset_info[dataset_id] = info
        self._add_to_cache(dataset_id, lazy_data)

        # Set as active if first dataset
        if self._active_dataset_id is None:
            self._active_dataset_id = dataset_id
            self.active_dataset_changed.emit(dataset_id)

        logger.info(f"Dataset added: {name} ({dataset_id[:8]}...) - "
                   f"{info.n_traces} traces, {info.n_ensembles} ensembles")

        # Emit signal
        self.dataset_added.emit(dataset_id, info.to_dict())

        return dataset_id

    def remove_dataset(self, dataset_id: str) -> bool:
        """
        Remove a dataset from the navigator.

        Args:
            dataset_id: Dataset identifier to remove

        Returns:
            True if dataset was removed, False if not found

        Emits:
            dataset_removed: With dataset_id
            active_dataset_changed: If active dataset was removed
        """
        if dataset_id not in self._dataset_info:
            logger.warning(f"Dataset not found for removal: {dataset_id[:8]}...")
            return False

        # Get info before removal
        info = self._dataset_info[dataset_id]
        name = info.name

        # Remove from registry and cache
        del self._dataset_info[dataset_id]
        if dataset_id in self._loaded_data:
            del self._loaded_data[dataset_id]

        logger.info(f"Dataset removed: {name} ({dataset_id[:8]}...)")

        # Emit removal signal
        self.dataset_removed.emit(dataset_id)

        # Handle active dataset removal
        if self._active_dataset_id == dataset_id:
            # Switch to most recently added dataset, or None
            if self._dataset_info:
                new_active = next(reversed(self._dataset_info))
                self._active_dataset_id = new_active
                self.active_dataset_changed.emit(new_active)
                logger.info(f"Active dataset switched to: {new_active[:8]}...")
            else:
                self._active_dataset_id = None
                self.active_dataset_changed.emit('')
                logger.info("No active dataset")

        return True

    def get_dataset(self, dataset_id: str) -> Optional[LazySeismicData]:
        """
        Get LazySeismicData for a dataset.

        If the dataset is not in the cache, attempts to reload from storage path.

        Args:
            dataset_id: Dataset identifier

        Returns:
            LazySeismicData instance or None if not found
        """
        if dataset_id not in self._dataset_info:
            return None

        # Check cache first
        if dataset_id in self._loaded_data:
            # Move to end (most recently used)
            self._loaded_data.move_to_end(dataset_id)
            return self._loaded_data[dataset_id]

        # Try to reload from storage path
        info = self._dataset_info[dataset_id]
        if info.storage_path.exists():
            try:
                lazy_data = LazySeismicData.from_storage_dir(str(info.storage_path))
                self._add_to_cache(dataset_id, lazy_data)
                logger.info(f"Dataset reloaded from cache: {info.name}")
                return lazy_data
            except Exception as e:
                logger.error(f"Failed to reload dataset {info.name}: {e}")
                return None
        else:
            logger.warning(f"Storage path not found for dataset {info.name}: {info.storage_path}")
            return None

    def set_active_dataset(self, dataset_id: str) -> bool:
        """
        Set the active dataset.

        Args:
            dataset_id: Dataset identifier to make active

        Returns:
            True if dataset was activated, False if not found

        Emits:
            active_dataset_changed: With new dataset_id
        """
        if dataset_id not in self._dataset_info:
            logger.warning(f"Cannot activate unknown dataset: {dataset_id[:8]}...")
            return False

        if self._active_dataset_id == dataset_id:
            return True  # Already active

        old_active = self._active_dataset_id
        self._active_dataset_id = dataset_id

        info = self._dataset_info[dataset_id]
        logger.info(f"Active dataset changed: {info.name} ({dataset_id[:8]}...)")

        self.active_dataset_changed.emit(dataset_id)
        return True

    def get_active_dataset(self) -> Optional[LazySeismicData]:
        """
        Get the currently active LazySeismicData.

        Returns:
            LazySeismicData for active dataset, or None if no active dataset
        """
        if self._active_dataset_id is None:
            return None
        return self.get_dataset(self._active_dataset_id)

    def get_active_dataset_id(self) -> Optional[str]:
        """
        Get the ID of the currently active dataset.

        Returns:
            Active dataset ID or None
        """
        return self._active_dataset_id

    def get_active_dataset_info(self) -> Optional[DatasetInfo]:
        """
        Get info for the currently active dataset.

        Returns:
            DatasetInfo for active dataset, or None
        """
        if self._active_dataset_id is None:
            return None
        return self._dataset_info.get(self._active_dataset_id)

    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        List all registered datasets with their info.

        Returns:
            List of dataset info dictionaries, ordered by load time
        """
        return [info.to_dict() for info in self._dataset_info.values()]

    def get_dataset_info(self, dataset_id: str) -> Optional[DatasetInfo]:
        """
        Get metadata for a specific dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            DatasetInfo or None if not found
        """
        return self._dataset_info.get(dataset_id)

    def get_dataset_count(self) -> int:
        """Get number of registered datasets."""
        return len(self._dataset_info)

    def has_datasets(self) -> bool:
        """Check if any datasets are loaded."""
        return len(self._dataset_info) > 0

    def clear_all(self) -> None:
        """
        Remove all datasets.

        Emits:
            datasets_cleared
            active_dataset_changed: With empty string
        """
        count = len(self._dataset_info)
        self._dataset_info.clear()
        self._loaded_data.clear()
        self._active_dataset_id = None

        logger.info(f"All datasets cleared ({count} removed)")

        self.datasets_cleared.emit()
        self.active_dataset_changed.emit('')

    def get_dataset_by_name(self, name: str) -> Optional[str]:
        """
        Find dataset ID by name.

        Args:
            name: Dataset name to search for

        Returns:
            Dataset ID or None if not found
        """
        for dataset_id, info in self._dataset_info.items():
            if info.name == name:
                return dataset_id
        return None

    def get_dataset_by_path(self, source_path: Path) -> Optional[str]:
        """
        Find dataset ID by source path.

        Args:
            source_path: Source file path to search for

        Returns:
            Dataset ID or None if not found
        """
        source_path = Path(source_path).resolve()
        for dataset_id, info in self._dataset_info.items():
            if info.source_path.resolve() == source_path:
                return dataset_id
        return None

    def update_dataset_name(self, dataset_id: str, new_name: str) -> bool:
        """
        Update the display name for a dataset.

        Args:
            dataset_id: Dataset identifier
            new_name: New display name

        Returns:
            True if updated, False if not found
        """
        if dataset_id not in self._dataset_info:
            return False

        self._dataset_info[dataset_id].name = new_name
        logger.info(f"Dataset renamed to: {new_name}")
        return True

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about loaded datasets.

        Returns:
            Dictionary with:
                - total_datasets: Number of registered datasets
                - cached_datasets: Number of datasets in memory
                - active_dataset_id: Current active dataset ID
                - total_traces: Sum of traces across all datasets
                - total_ensembles: Sum of ensembles across all datasets
        """
        total_traces = sum(info.n_traces for info in self._dataset_info.values())
        total_ensembles = sum(info.n_ensembles for info in self._dataset_info.values())

        return {
            'total_datasets': len(self._dataset_info),
            'cached_datasets': len(self._loaded_data),
            'max_cached': self._max_cached_datasets,
            'active_dataset_id': self._active_dataset_id,
            'total_traces': total_traces,
            'total_ensembles': total_ensembles,
        }

    def _add_to_cache(self, dataset_id: str, lazy_data: LazySeismicData) -> None:
        """
        Add dataset to LRU cache, evicting oldest if needed.

        Args:
            dataset_id: Dataset identifier
            lazy_data: LazySeismicData to cache
        """
        # Evict oldest if cache full (but never evict active dataset)
        while len(self._loaded_data) >= self._max_cached_datasets:
            # Find oldest non-active dataset
            for old_id in self._loaded_data:
                if old_id != self._active_dataset_id:
                    del self._loaded_data[old_id]
                    logger.debug(f"Evicted dataset from cache: {old_id[:8]}...")
                    break
            else:
                # All cached datasets are active (shouldn't happen with max > 1)
                break

        self._loaded_data[dataset_id] = lazy_data

    def to_serializable(self) -> Dict[str, Any]:
        """
        Convert navigator state to serializable dictionary.

        Returns:
            Dictionary suitable for JSON serialization
        """
        return {
            'datasets': [info.to_dict() for info in self._dataset_info.values()],
            'active_dataset_id': self._active_dataset_id,
            'max_cached_datasets': self._max_cached_datasets,
        }

    def restore_from_serialized(self, data: Dict[str, Any]) -> int:
        """
        Restore navigator state from serialized data.

        Note: This only restores dataset info, not the actual data.
        Datasets will be loaded on-demand when accessed.

        Args:
            data: Dictionary from to_serializable()

        Returns:
            Number of datasets successfully restored
        """
        restored = 0

        for dataset_dict in data.get('datasets', []):
            try:
                info = DatasetInfo.from_dict(dataset_dict)

                # Verify storage path exists
                if not info.storage_path.exists():
                    logger.warning(f"Skipping dataset {info.name}: storage path not found")
                    continue

                # Add to registry (data will be loaded on-demand)
                self._dataset_info[info.dataset_id] = info
                restored += 1
                logger.info(f"Restored dataset info: {info.name}")

            except Exception as e:
                logger.error(f"Failed to restore dataset: {e}")

        # Restore active dataset
        active_id = data.get('active_dataset_id')
        if active_id and active_id in self._dataset_info:
            self._active_dataset_id = active_id
        elif self._dataset_info:
            # Default to first dataset if saved active is invalid
            self._active_dataset_id = next(iter(self._dataset_info))

        if self._active_dataset_id:
            self.active_dataset_changed.emit(self._active_dataset_id)

        logger.info(f"Restored {restored} datasets from saved state")
        return restored

    def __repr__(self) -> str:
        active_name = None
        if self._active_dataset_id and self._active_dataset_id in self._dataset_info:
            active_name = self._dataset_info[self._active_dataset_id].name

        return (f"DatasetNavigator(datasets={len(self._dataset_info)}, "
                f"cached={len(self._loaded_data)}, "
                f"active='{active_name}')")
