"""
Global application settings and preferences.
Uses QSettings for persistent storage across sessions.
"""
import os
from PyQt6.QtCore import QSettings, pyqtSignal, QObject
from typing import Optional


class AppSettings(QObject):
    """
    Singleton class for managing global application settings.

    Settings include:
    - Spatial units (meters/feet)
    - Other app-wide preferences

    Settings are automatically persisted using QSettings.
    """

    # Signals emitted when settings change
    spatial_units_changed = pyqtSignal(str)  # Emits 'meters' or 'feet'

    _instance: Optional['AppSettings'] = None

    # Available spatial units
    METERS = 'meters'
    FEET = 'feet'
    VALID_UNITS = [METERS, FEET]

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

        super().__init__()

        # Default values (defined first for fallback)
        self._defaults = {
            'spatial_units': self.METERS,
            'window_geometry': None,
            'recent_files': [],
        }

        # Check if we should disable QSettings (problematic environments)
        # If QT_XCB_GL_INTEGRATION=none, we're likely in a crash-prone environment
        disable_qsettings = os.environ.get('QT_XCB_GL_INTEGRATION') == 'none'

        if disable_qsettings:
            # Skip QSettings entirely - use in-memory only
            import logging
            logger = logging.getLogger(__name__)
            logger.info("QSettings disabled (problematic environment detected), using in-memory defaults")
            self.settings = None
            self._settings_available = False
        else:
            # Initialize QSettings (may still fail on some systems)
            try:
                # Force portable INI format instead of native registry/D-Bus
                self.settings = QSettings(
                    QSettings.Format.IniFormat,
                    QSettings.Scope.UserScope,
                    'SeismicDenoise',
                    'DenoiseApp'
                )
                self._settings_available = True
            except Exception as e:
                # QSettings may crash/fail on some systems (WSL, display issues)
                # Fall back to in-memory defaults
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"QSettings initialization failed, using defaults: {e}")
                self.settings = None
                self._settings_available = False

        self._initialized = True

    def get_spatial_units(self) -> str:
        """
        Get the current spatial units setting.

        Returns:
            'meters' or 'feet'
        """
        if not self._settings_available or self.settings is None:
            return self._defaults['spatial_units']

        units = self.settings.value('spatial_units', self._defaults['spatial_units'])
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

        old_units = self.get_spatial_units()
        if old_units != units:
            if self._settings_available and self.settings is not None:
                self.settings.setValue('spatial_units', units)
            # Always emit signal and update in-memory default
            self._defaults['spatial_units'] = units
            self.spatial_units_changed.emit(units)

    def is_meters(self) -> bool:
        """Check if current units are meters."""
        return self.get_spatial_units() == self.METERS

    def is_feet(self) -> bool:
        """Check if current units are feet."""
        return self.get_spatial_units() == self.FEET

    def get_recent_files(self) -> list:
        """Get list of recently opened files."""
        if not self._settings_available or self.settings is None:
            return self._defaults['recent_files']
        return self.settings.value('recent_files', self._defaults['recent_files'])

    def add_recent_file(self, filepath: str):
        """Add file to recent files list."""
        recent = self.get_recent_files()
        if filepath in recent:
            recent.remove(filepath)
        recent.insert(0, filepath)
        recent = recent[:10]  # Keep only 10 most recent
        if self._settings_available and self.settings is not None:
            self.settings.setValue('recent_files', recent)

    def get_window_geometry(self):
        """Get saved window geometry."""
        if not self._settings_available or self.settings is None:
            return self._defaults['window_geometry']
        return self.settings.value('window_geometry', self._defaults['window_geometry'])

    def set_window_geometry(self, geometry):
        """Save window geometry."""
        if self._settings_available and self.settings is not None:
            self.settings.setValue('window_geometry', geometry)

    def reset_to_defaults(self):
        """Reset all settings to default values."""
        if self._settings_available and self.settings is not None:
            for key, value in self._defaults.items():
                self.settings.setValue(key, value)

        # Emit signals for changed settings
        self.spatial_units_changed.emit(self.get_spatial_units())

    def __repr__(self) -> str:
        return f"AppSettings(spatial_units={self.get_spatial_units()})"


# Global singleton instance
def get_settings() -> AppSettings:
    """Get the global settings instance."""
    return AppSettings()
