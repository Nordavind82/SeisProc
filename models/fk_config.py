"""
FK Filter Configuration Management

Handles saving, loading, and managing FK filter configurations.
Supports both metric (m/s) and imperial (ft/s) velocity units.
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from utils.unit_conversion import (
    convert_velocity_to_metric,
    convert_velocity_from_metric,
    METERS_TO_FEET,
    FEET_TO_METERS
)


@dataclass
class SubGather:
    """
    Represents a sub-gather within a larger gather.

    Sub-gathers are created by splitting a gather based on header value changes
    (e.g., when ReceiverLine changes within a common shot gather).
    """
    sub_id: int              # 0-based index within parent gather
    start_trace: int         # Start trace index (absolute within gather)
    end_trace: int           # End trace index (absolute, inclusive)
    n_traces: int            # Number of traces in this sub-gather
    boundary_header: str     # Header used for boundary (e.g., "ReceiverLine")
    boundary_value: Any      # Value of boundary header for this sub-gather
    description: str         # Human-readable description

    def __repr__(self) -> str:
        return f"SubGather({self.sub_id}: {self.description}, {self.n_traces} traces)"


@dataclass
class FKFilterConfig:
    """
    FK filter configuration.

    Stores all parameters needed to apply an FK filter, plus metadata
    for organization and tracking.
    """
    name: str
    filter_type: str  # 'velocity_fan', 'dip', 'polygon' (currently only velocity_fan)
    v_min: float  # Minimum velocity (m/s)
    v_max: float  # Maximum velocity (m/s)
    taper_width: float  # Taper width (m/s)
    mode: str  # 'pass' or 'reject'
    created: str  # ISO format timestamp
    created_on_gather: Optional[int] = None
    description: str = ""
    author: str = "user"

    # Sub-gather settings
    use_subgathers: bool = False
    boundary_header: Optional[str] = None  # e.g., "ReceiverLine"

    # AGC settings
    apply_agc: bool = False
    agc_window_ms: float = 500.0

    # Unit settings (velocities stored in these units)
    coordinate_units: str = 'meters'  # 'meters' or 'feet'

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'FKFilterConfig':
        """Create from dictionary."""
        return cls(**data)

    def to_processor_params(self, trace_spacing: float) -> Dict:
        """
        Convert to processor parameters.

        Args:
            trace_spacing: Trace spacing in coordinate units (from gather)

        Returns:
            Dictionary of parameters for FKFilter processor
        """
        return {
            'v_min': self.v_min,
            'v_max': self.v_max,
            'taper_width': self.taper_width,
            'mode': self.mode,
            'trace_spacing': trace_spacing,
            'coordinate_units': self.coordinate_units
        }

    def get_summary(self) -> str:
        """Get one-line summary of configuration."""
        mode_str = "Pass" if self.mode == 'pass' else "Reject"
        unit_abbrev = "ft/s" if self.coordinate_units == 'feet' else "m/s"
        return f"{mode_str}: {self.v_min:.0f}-{self.v_max:.0f} {unit_abbrev}, Taper: {self.taper_width:.0f} {unit_abbrev}"


class FKConfigManager:
    """
    Manages FK filter configurations.

    Handles loading, saving, and organizing multiple FK filter configurations.
    Configurations are stored as JSON files in a configs directory.
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration manager.

        Args:
            config_dir: Directory to store configurations.
                       Defaults to ~/.denoise_app/fk_configs/
        """
        if config_dir is None:
            config_dir = Path.home() / '.denoise_app' / 'fk_configs'

        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.configs: List[FKFilterConfig] = []
        self._load_all_configs()

    def _load_all_configs(self):
        """Load all configurations from config directory."""
        self.configs = []

        # Load from JSON files
        for config_file in self.config_dir.glob('*.json'):
            try:
                with open(config_file, 'r') as f:
                    data = json.load(f)
                    config = FKFilterConfig.from_dict(data)
                    self.configs.append(config)
            except Exception as e:
                print(f"Warning: Failed to load config {config_file}: {e}")

        # Sort by creation time (newest first)
        self.configs.sort(key=lambda c: c.created, reverse=True)

    def save_config(self, config: FKFilterConfig) -> Path:
        """
        Save configuration to file.

        Args:
            config: Configuration to save

        Returns:
            Path to saved file
        """
        # Generate filename from name (sanitized)
        safe_name = "".join(c for c in config.name if c.isalnum() or c in (' ', '_', '-'))
        safe_name = safe_name.replace(' ', '_')
        filename = f"{safe_name}.json"
        filepath = self.config_dir / filename

        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)

        # Update in-memory list
        # Remove existing config with same name if present
        self.configs = [c for c in self.configs if c.name != config.name]
        self.configs.insert(0, config)  # Add to front (newest)

        return filepath

    def delete_config(self, config_name: str) -> bool:
        """
        Delete configuration.

        Args:
            config_name: Name of configuration to delete

        Returns:
            True if deleted, False if not found
        """
        # Find config
        config = self.get_config(config_name)
        if config is None:
            return False

        # Delete file
        safe_name = "".join(c for c in config_name if c.isalnum() or c in (' ', '_', '-'))
        safe_name = safe_name.replace(' ', '_')
        filepath = self.config_dir / f"{safe_name}.json"

        if filepath.exists():
            filepath.unlink()

        # Remove from list
        self.configs = [c for c in self.configs if c.name != config_name]

        return True

    def get_config(self, name: str) -> Optional[FKFilterConfig]:
        """
        Get configuration by name.

        Args:
            name: Configuration name

        Returns:
            Configuration or None if not found
        """
        for config in self.configs:
            if config.name == name:
                return config
        return None

    def get_all_configs(self) -> List[FKFilterConfig]:
        """Get all configurations, sorted by creation time (newest first)."""
        return self.configs.copy()

    def export_config(self, config_name: str, export_path: Path):
        """
        Export configuration to external file.

        Args:
            config_name: Name of configuration to export
            export_path: Path to export to

        Raises:
            ValueError: If configuration not found
        """
        config = self.get_config(config_name)
        if config is None:
            raise ValueError(f"Configuration '{config_name}' not found")

        with open(export_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)

    def import_config(self, import_path: Path) -> FKFilterConfig:
        """
        Import configuration from external file.

        Args:
            import_path: Path to import from

        Returns:
            Imported configuration

        Raises:
            ValueError: If file is invalid
        """
        with open(import_path, 'r') as f:
            data = json.load(f)

        config = FKFilterConfig.from_dict(data)

        # Check if name already exists
        if self.get_config(config.name) is not None:
            # Append timestamp to make unique
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config.name = f"{config.name}_{timestamp}"

        # Save imported config
        self.save_config(config)

        return config

    def create_from_preset(self, preset_name: str, custom_name: Optional[str] = None) -> FKFilterConfig:
        """
        Create configuration from preset.

        Args:
            preset_name: Name of preset (from get_fk_filter_presets)
            custom_name: Optional custom name (defaults to preset name)

        Returns:
            New configuration

        Raises:
            ValueError: If preset not found
        """
        from processors.fk_filter import get_fk_filter_presets

        presets = get_fk_filter_presets()
        if preset_name not in presets:
            raise ValueError(f"Preset '{preset_name}' not found")

        preset = presets[preset_name]

        config = FKFilterConfig(
            name=custom_name or preset_name,
            filter_type='velocity_fan',
            v_min=preset['v_min'],
            v_max=preset['v_max'],
            taper_width=preset['taper_width'],
            mode=preset['mode'],
            created=datetime.now().isoformat(),
            description=preset['description']
        )

        return config


def create_default_configs() -> List[FKFilterConfig]:
    """
    Create default FK filter configurations.

    Returns:
        List of default configurations (not yet saved)
    """
    from processors.fk_filter import get_fk_filter_presets

    presets = get_fk_filter_presets()
    configs = []

    for name, params in presets.items():
        config = FKFilterConfig(
            name=name,
            filter_type='velocity_fan',
            v_min=params['v_min'],
            v_max=params['v_max'],
            taper_width=params['taper_width'],
            mode=params['mode'],
            created=datetime.now().isoformat(),
            description=params['description']
        )
        configs.append(config)

    return configs
