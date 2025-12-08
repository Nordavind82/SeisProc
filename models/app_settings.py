"""
Global application settings and preferences.
Uses JSON file for persistent storage across sessions.

Includes:
- Spatial units (meters/feet)
- Dataset management (loaded datasets, active dataset)
- Session state (viewport, gain, colormap)
- Window geometry and recent files

Note: This class does NOT inherit from QObject to avoid segfaults
in environments where QApplication may not be initialized before import.
Uses JSON file storage instead of QSettings for reliability.
"""
import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable

# Set up module logger
logger = logging.getLogger(__name__)


class AppSettings:
    """
    Singleton class for managing global application settings.

    Settings include:
    - Spatial units (meters/feet)
    - Dataset management (loaded datasets, active dataset)
    - Session state (viewport, gain, colormap, current gather)
    - Window geometry and layout
    - Recent files

    Settings are automatically persisted to a JSON file (~/.seisproc/settings.json).
    """

    _instance: Optional['AppSettings'] = None

    # Available spatial units
    METERS = 'meters'
    FEET = 'feet'
    VALID_UNITS = [METERS, FEET]

    # Settings file location
    SETTINGS_DIR = Path.home() / '.seisproc'
    SETTINGS_FILE = SETTINGS_DIR / 'settings.json'

    def __new__(cls):
        """Singleton pattern - only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize settings (only once due to singleton)."""
        if self._initialized:
            return

        # Default values
        self._defaults = {
            'spatial_units': self.METERS,
            'window_geometry': None,
            'recent_files': [],
            # Dataset management
            'loaded_datasets': [],           # List of dataset info dicts
            'active_dataset_id': None,       # UUID of active dataset
            'dataset_cache_limit': 3,        # Max datasets in memory
            'auto_load_last_dataset': True,  # Load last dataset on startup
            # Session state
            'session_state': {
                'viewport': {
                    'time_min': 0.0,
                    'time_max': 1000.0,
                    'trace_min': 0.0,
                    'trace_max': 100.0,
                },
                'gain': 1.0,
                'colormap': 'seismic',
                'interpolation': 'bilinear',
                'current_gather_id': 0,
                'sort_keys': [],
            },
            # Storage settings
            'storage_directory': None,
            'gather_cache_limit': 5,
            'remember_window_geometry': True,
            'max_recent_files': 10,
            # GPU/Processing settings
            'gpu_enabled': True,              # Use GPU if available
            'gpu_device_preference': 'auto',  # 'auto', 'cuda', 'mps', 'cpu'
            'processing_workers_auto': True,  # Auto-calculate workers
            'processing_workers': 1,          # Manual worker count (when auto=False)
            'gpu_memory_limit_percent': 70,   # Max GPU memory to use (%)
            'cpu_workers_auto': True,         # Auto-calculate CPU workers
            'cpu_workers': 4,                 # Manual CPU worker count
        }

        # Current settings (loaded from file or defaults)
        self._settings: Dict[str, Any] = {}

        # Ensure settings directory exists
        self._ensure_settings_dir()

        # Load settings from file
        self._load_settings()

        self._initialized = True
        logger.info(f"AppSettings initialized from {self.SETTINGS_FILE}")

    def _ensure_settings_dir(self):
        """Ensure the settings directory exists."""
        try:
            self.SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create settings directory: {e}")

    def _load_settings(self):
        """Load settings from JSON file."""
        if self.SETTINGS_FILE.exists():
            try:
                with open(self.SETTINGS_FILE, 'r', encoding='utf-8') as f:
                    self._settings = json.load(f)
                logger.debug(f"Loaded settings from {self.SETTINGS_FILE}")
            except Exception as e:
                logger.warning(f"Could not load settings file: {e}")
                self._settings = {}
        else:
            self._settings = {}
            logger.debug("No settings file found, using defaults")

    def _save_settings(self):
        """Save settings to JSON file."""
        try:
            self._ensure_settings_dir()
            with open(self.SETTINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self._settings, f, indent=2, default=str)
            logger.debug(f"Saved settings to {self.SETTINGS_FILE}")
        except Exception as e:
            logger.error(f"Could not save settings: {e}")

    def _get(self, key: str, default=None):
        """Get a setting value, falling back to defaults."""
        if default is None:
            default = self._defaults.get(key)
        return self._settings.get(key, default)

    def _set(self, key: str, value: Any, save: bool = True):
        """Set a setting value and optionally save to file."""
        self._settings[key] = value
        if save:
            self._save_settings()

    def get_spatial_units(self) -> str:
        """
        Get the current spatial units setting.

        Returns:
            'meters' or 'feet'
        """
        units = self._get('spatial_units', self.METERS)
        if units not in self.VALID_UNITS:
            units = self.METERS
        return units

    def set_spatial_units(self, units: str):
        """
        Set the spatial units.

        Args:
            units: 'meters' or 'feet'
        """
        if units not in self.VALID_UNITS:
            raise ValueError(f"Invalid units: {units}. Must be 'meters' or 'feet'")
        self._set('spatial_units', units)

    def is_meters(self) -> bool:
        """Check if current units are meters."""
        return self.get_spatial_units() == self.METERS

    def is_feet(self) -> bool:
        """Check if current units are feet."""
        return self.get_spatial_units() == self.FEET

    def get_recent_files(self) -> list:
        """Get list of recently opened files."""
        return self._get('recent_files', [])

    def add_recent_file(self, filepath: str):
        """Add file to recent files list."""
        recent = self.get_recent_files().copy()
        if filepath in recent:
            recent.remove(filepath)
        recent.insert(0, filepath)
        max_files = self.get_max_recent_files()
        recent = recent[:max_files]
        self._set('recent_files', recent)

    def get_window_geometry(self):
        """Get saved window geometry."""
        return self._get('window_geometry')

    def set_window_geometry(self, geometry):
        """Save window geometry."""
        self._set('window_geometry', geometry)

    def reset_to_defaults(self):
        """Reset all settings to default values."""
        self._settings = self._defaults.copy()
        self._save_settings()
        logger.info("Settings reset to defaults")

    # =========================================================================
    # Dataset Management Methods
    # =========================================================================

    def get_loaded_datasets(self) -> List[Dict[str, Any]]:
        """
        Get list of previously loaded datasets.

        Returns:
            List of dataset info dictionaries with keys:
                - dataset_id: UUID string
                - name: Display name
                - source_path: Original file path
                - storage_path: Zarr storage directory
                - loaded_at: ISO timestamp
        """
        datasets = self._get('loaded_datasets', [])
        return datasets if isinstance(datasets, list) else []

    def save_loaded_datasets(self, datasets: List[Dict[str, Any]]) -> None:
        """
        Save list of loaded datasets.

        Args:
            datasets: List of dataset info dictionaries
        """
        self._set('loaded_datasets', datasets)
        logger.debug(f"Saved {len(datasets)} datasets to settings")

    def add_loaded_dataset(self, dataset_info: Dict[str, Any]) -> None:
        """
        Add a dataset to the loaded datasets list.

        Args:
            dataset_info: Dataset info dictionary
        """
        datasets = self.get_loaded_datasets().copy()

        # Remove if already exists (by ID or path)
        dataset_id = dataset_info.get('dataset_id')
        source_path = dataset_info.get('source_path')

        datasets = [d for d in datasets
                   if d.get('dataset_id') != dataset_id
                   and d.get('source_path') != source_path]

        # Add to front of list
        datasets.insert(0, dataset_info)

        # Keep only last 20 datasets
        datasets = datasets[:20]

        self.save_loaded_datasets(datasets)

    def remove_loaded_dataset(self, dataset_id: str) -> bool:
        """
        Remove a dataset from the loaded datasets list.

        Args:
            dataset_id: Dataset UUID to remove

        Returns:
            True if removed, False if not found
        """
        datasets = self.get_loaded_datasets().copy()
        original_len = len(datasets)

        datasets = [d for d in datasets if d.get('dataset_id') != dataset_id]

        if len(datasets) < original_len:
            self.save_loaded_datasets(datasets)
            return True
        return False

    def get_active_dataset_id(self) -> Optional[str]:
        """
        Get the ID of the last active dataset.

        Returns:
            Dataset UUID string or None
        """
        return self._get('active_dataset_id')

    def set_active_dataset_id(self, dataset_id: Optional[str]) -> None:
        """
        Set the active dataset ID.

        Args:
            dataset_id: Dataset UUID string or None
        """
        self._set('active_dataset_id', dataset_id)

    def get_dataset_cache_limit(self) -> int:
        """Get maximum number of datasets to keep in memory."""
        value = self._get('dataset_cache_limit', 3)
        try:
            return int(value)
        except (TypeError, ValueError):
            return 3

    def set_dataset_cache_limit(self, limit: int) -> None:
        """Set maximum number of datasets to keep in memory."""
        limit = max(1, min(10, limit))  # Clamp to 1-10
        self._set('dataset_cache_limit', limit)

    def get_auto_load_last_dataset(self) -> bool:
        """Check if last dataset should be auto-loaded on startup."""
        value = self._get('auto_load_last_dataset', True)
        if isinstance(value, bool):
            return value
        return str(value).lower() in ('true', '1', 'yes')

    def set_auto_load_last_dataset(self, enabled: bool) -> None:
        """Set whether to auto-load last dataset on startup."""
        self._set('auto_load_last_dataset', enabled)

    # =========================================================================
    # Session State Methods
    # =========================================================================

    def get_session_state(self) -> Dict[str, Any]:
        """
        Get the saved session state.

        Returns:
            Dictionary with viewport, gain, colormap, etc.
        """
        default_state = self._defaults['session_state']
        state = self._get('session_state', default_state)

        if isinstance(state, dict):
            # Merge with defaults to ensure all keys exist
            return {**default_state, **state}
        return default_state.copy()

    def save_session_state(self, state: Dict[str, Any]) -> None:
        """
        Save session state for restoration on next launch.

        Args:
            state: Dictionary with session state:
                - viewport: {time_min, time_max, trace_min, trace_max}
                - gain: float
                - colormap: str
                - interpolation: str
                - current_gather_id: int
                - sort_keys: list
        """
        self._set('session_state', state)
        logger.debug("Session state saved")

    def save_viewport_state(self, time_min: float, time_max: float,
                           trace_min: float, trace_max: float) -> None:
        """
        Save viewport limits to session state.

        Args:
            time_min, time_max: Time range in milliseconds
            trace_min, trace_max: Trace range
        """
        state = self.get_session_state()
        state['viewport'] = {
            'time_min': time_min,
            'time_max': time_max,
            'trace_min': trace_min,
            'trace_max': trace_max,
        }
        self.save_session_state(state)

    def save_display_state(self, gain: float = None, colormap: str = None,
                          interpolation: str = None) -> None:
        """
        Save display settings to session state.

        Args:
            gain: Optional gain value
            colormap: Optional colormap name
            interpolation: Optional interpolation mode
        """
        state = self.get_session_state()
        if gain is not None:
            state['gain'] = gain
        if colormap is not None:
            state['colormap'] = colormap
        if interpolation is not None:
            state['interpolation'] = interpolation
        self.save_session_state(state)

    def save_navigation_state(self, current_gather_id: int = None,
                             sort_keys: List[str] = None) -> None:
        """
        Save navigation state to session.

        Args:
            current_gather_id: Current gather index
            sort_keys: List of header keys for sorting
        """
        state = self.get_session_state()
        if current_gather_id is not None:
            state['current_gather_id'] = current_gather_id
        if sort_keys is not None:
            state['sort_keys'] = sort_keys
        self.save_session_state(state)

    def restore_session(self) -> Dict[str, Any]:
        """
        Restore complete session state and emit signal.

        Returns:
            Session state dictionary
        """
        state = self.get_session_state()
        logger.info("Session state restored")
        return state

    def clear_session_state(self) -> None:
        """Clear saved session state to defaults."""
        self.save_session_state(self._defaults['session_state'].copy())
        logger.info("Session state cleared")

    # =========================================================================
    # Storage Directory Settings
    # =========================================================================

    def get_default_storage_directory(self) -> Path:
        """Get the default storage directory path."""
        return Path.home() / '.seisproc' / 'data'

    def get_storage_directory(self) -> Optional[Path]:
        """
        Get the configured storage directory for Zarr/Parquet data.

        Returns:
            Path to storage directory, or None if using default
        """
        value = self._get('storage_directory')
        if value:
            return Path(value)
        return None

    def set_storage_directory(self, path: Optional[str]) -> None:
        """
        Set the storage directory for Zarr/Parquet data.

        Args:
            path: Directory path, or None to use default
        """
        self._set('storage_directory', str(path) if path else None)
        logger.info(f"Storage directory set to: {path or 'default'}")

    def get_effective_storage_directory(self) -> Path:
        """
        Get the effective storage directory (configured or default).

        Creates the directory if it doesn't exist.

        Returns:
            Path to storage directory
        """
        custom = self.get_storage_directory()
        if custom:
            path = Path(custom)
        else:
            path = self.get_default_storage_directory()

        # Ensure directory exists
        path.mkdir(parents=True, exist_ok=True)
        return path

    # =========================================================================
    # Cache Settings
    # =========================================================================

    def get_gather_cache_limit(self) -> int:
        """Get maximum number of gathers to cache per dataset."""
        value = self._get('gather_cache_limit', 5)
        try:
            return int(value)
        except (TypeError, ValueError):
            return 5

    def set_gather_cache_limit(self, limit: int) -> None:
        """Set maximum number of gathers to cache per dataset."""
        limit = max(1, min(20, limit))
        self._set('gather_cache_limit', limit)

    # =========================================================================
    # Window and UI Settings
    # =========================================================================

    def get_remember_window_geometry(self) -> bool:
        """Check if window geometry should be restored on startup."""
        value = self._get('remember_window_geometry', True)
        if isinstance(value, bool):
            return value
        return str(value).lower() in ('true', '1', 'yes')

    def set_remember_window_geometry(self, enabled: bool) -> None:
        """Set whether to remember window geometry."""
        self._set('remember_window_geometry', enabled)

    def get_max_recent_files(self) -> int:
        """Get maximum number of recent files to store."""
        value = self._get('max_recent_files', 10)
        try:
            return int(value)
        except (TypeError, ValueError):
            return 10

    def set_max_recent_files(self, count: int) -> None:
        """Set maximum number of recent files."""
        count = max(5, min(50, count))
        self._set('max_recent_files', count)

    def clear_recent_files(self) -> None:
        """Clear the recent files list."""
        self._set('recent_files', [])
        logger.info("Recent files cleared")

    # =========================================================================
    # GPU/Processing Settings
    # =========================================================================

    def get_gpu_enabled(self) -> bool:
        """Check if GPU acceleration is enabled."""
        value = self._get('gpu_enabled', True)
        return bool(value) if isinstance(value, bool) else str(value).lower() in ('true', '1', 'yes')

    def set_gpu_enabled(self, enabled: bool) -> None:
        """Enable or disable GPU acceleration."""
        self._set('gpu_enabled', enabled)
        logger.info(f"GPU acceleration {'enabled' if enabled else 'disabled'}")

    def get_gpu_device_preference(self) -> str:
        """Get preferred GPU device ('auto', 'cuda', 'mps', 'cpu')."""
        value = self._get('gpu_device_preference', 'auto')
        if value not in ('auto', 'cuda', 'mps', 'cpu'):
            return 'auto'
        return value

    def set_gpu_device_preference(self, device: str) -> None:
        """Set preferred GPU device."""
        if device not in ('auto', 'cuda', 'mps', 'cpu'):
            raise ValueError(f"Invalid device: {device}. Must be 'auto', 'cuda', 'mps', or 'cpu'")
        self._set('gpu_device_preference', device)

    def get_processing_workers_auto(self) -> bool:
        """Check if processing workers should be auto-calculated."""
        value = self._get('processing_workers_auto', True)
        return bool(value) if isinstance(value, bool) else str(value).lower() in ('true', '1', 'yes')

    def set_processing_workers_auto(self, auto: bool) -> None:
        """Set whether to auto-calculate processing workers."""
        self._set('processing_workers_auto', auto)

    def get_processing_workers(self) -> int:
        """Get manual processing worker count."""
        value = self._get('processing_workers', 1)
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 1

    def set_processing_workers(self, workers: int) -> None:
        """Set manual processing worker count."""
        self._set('processing_workers', max(1, workers))

    def get_gpu_memory_limit_percent(self) -> int:
        """Get GPU memory usage limit (percentage)."""
        value = self._get('gpu_memory_limit_percent', 70)
        try:
            return max(10, min(95, int(value)))
        except (TypeError, ValueError):
            return 70

    def set_gpu_memory_limit_percent(self, percent: int) -> None:
        """Set GPU memory usage limit."""
        self._set('gpu_memory_limit_percent', max(10, min(95, percent)))

    def get_cpu_workers_auto(self) -> bool:
        """Check if CPU workers should be auto-calculated."""
        value = self._get('cpu_workers_auto', True)
        return bool(value) if isinstance(value, bool) else str(value).lower() in ('true', '1', 'yes')

    def set_cpu_workers_auto(self, auto: bool) -> None:
        """Set whether to auto-calculate CPU workers."""
        self._set('cpu_workers_auto', auto)

    def get_cpu_workers(self) -> int:
        """Get manual CPU worker count."""
        value = self._get('cpu_workers', 4)
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 4

    def set_cpu_workers(self, workers: int) -> None:
        """Set manual CPU worker count."""
        self._set('cpu_workers', max(1, workers))

    def get_effective_workers(self, gpu_available: bool = False) -> int:
        """
        Get effective worker count based on settings and GPU availability.

        Args:
            gpu_available: Whether GPU is currently available

        Returns:
            Recommended worker count
        """
        import multiprocessing

        if self.get_gpu_enabled() and gpu_available:
            # GPU mode: fewer workers since GPU handles parallelism
            if self.get_processing_workers_auto():
                # Auto: 1 worker for GPU (GPU handles parallelism internally)
                return 1
            else:
                return self.get_processing_workers()
        else:
            # CPU mode: more workers for parallelism
            if self.get_cpu_workers_auto():
                # Auto: CPU cores - 1 (leave one for system)
                return max(1, multiprocessing.cpu_count() - 1)
            else:
                return self.get_cpu_workers()

    def get_recommended_gpu_workers(self) -> int:
        """Get recommended GPU worker count (always 1 for optimal GPU utilization)."""
        return 1

    def get_recommended_cpu_workers(self) -> int:
        """Get recommended CPU worker count based on system."""
        import multiprocessing
        return max(1, multiprocessing.cpu_count() - 1)

    def __repr__(self) -> str:
        datasets = self.get_loaded_datasets()
        return (f"AppSettings(spatial_units={self.get_spatial_units()}, "
                f"datasets={len(datasets)})")


# Global singleton instance
def get_settings() -> AppSettings:
    """Get the global settings instance."""
    return AppSettings()
