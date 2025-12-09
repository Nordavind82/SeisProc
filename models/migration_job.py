"""
Migration Job Configuration

Combines all components needed for a migration run:
- Velocity model
- Migration configuration
- Binning table
- Header mapping
- Input/output paths
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from pathlib import Path
import json
import logging
from datetime import datetime

from models.binning import BinningTable
from processors.migration.base import MigrationConfig, OutputGrid
from models.velocity import VelocityModel

logger = logging.getLogger(__name__)


@dataclass
class HeaderMapping:
    """Mapping from input headers to required fields."""
    source_x: Optional[str] = None
    source_y: Optional[str] = None
    receiver_x: Optional[str] = None
    receiver_y: Optional[str] = None
    offset: Optional[str] = None
    azimuth: Optional[str] = None
    inline: Optional[str] = None
    xline: Optional[str] = None
    cdp_x: Optional[str] = None
    cdp_y: Optional[str] = None

    def to_dict(self) -> Dict[str, Optional[str]]:
        """Convert to dictionary for indexer."""
        mapping = {}
        for field_name in [
            'source_x', 'source_y', 'receiver_x', 'receiver_y',
            'offset', 'azimuth', 'inline', 'xline', 'cdp_x', 'cdp_y'
        ]:
            value = getattr(self, field_name)
            if value is not None:
                mapping[field_name] = value
        return mapping

    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> 'HeaderMapping':
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class MigrationJobConfig:
    """
    Complete configuration for a migration job.

    Combines velocity, migration parameters, binning, and I/O paths.
    """
    # Required fields
    name: str
    input_file: str
    output_directory: str

    # Velocity
    velocity_v0: float = 2500.0  # m/s for constant velocity
    velocity_gradient: float = 0.0  # s^-1 for v(z) = v0 + gradient * z
    velocity_file: Optional[str] = None  # Path to velocity model file

    # Output grid
    time_min_ms: float = 0.0
    time_max_ms: float = 4000.0
    dt_ms: float = 4.0
    inline_min: int = 1
    inline_max: int = 100
    inline_step: int = 1
    xline_min: int = 1
    xline_max: int = 100
    xline_step: int = 1

    # Grid geometry
    origin_x: float = 0.0
    origin_y: float = 0.0
    inline_spacing: float = 25.0
    xline_spacing: float = 25.0

    # Migration parameters
    max_aperture_m: float = 3000.0
    max_angle_deg: float = 60.0
    max_offset_m: float = 10000.0
    taper_width_samples: int = 20
    near_offset_taper_m: float = 0.0
    anti_alias_filter: bool = True

    # Binning
    binning_preset: Optional[str] = None  # 'land_3d', 'marine', etc.
    binning_table: Optional[BinningTable] = None

    # Header mapping
    header_mapping: HeaderMapping = field(default_factory=HeaderMapping)

    # Processing options
    processing_mode: str = 'sequential'  # 'sequential' or 'parallel'
    max_workers: int = 1
    memory_limit_gb: float = 8.0
    enable_checkpointing: bool = True

    # Output options
    output_format: str = 'segy'  # 'segy', 'zarr', 'npy'
    create_stack_volume: bool = True
    create_fold_volume: bool = False
    compress_output: bool = False

    # Metadata
    created_at: Optional[str] = None
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and initialize after creation."""
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

    def validate(self) -> List[str]:
        """
        Validate job configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check input file
        if not Path(self.input_file).exists():
            errors.append(f"Input file does not exist: {self.input_file}")

        # Check output directory
        output_dir = Path(self.output_directory)
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create output directory: {e}")

        # Check velocity
        if self.velocity_v0 <= 0:
            errors.append(f"Invalid velocity v0: {self.velocity_v0}")

        # Check output grid
        if self.time_max_ms <= self.time_min_ms:
            errors.append("time_max_ms must be greater than time_min_ms")

        if self.dt_ms <= 0:
            errors.append("dt_ms must be positive")

        if self.inline_max < self.inline_min:
            errors.append("inline_max must be >= inline_min")

        if self.xline_max < self.xline_min:
            errors.append("xline_max must be >= xline_min")

        # Check migration parameters
        if self.max_aperture_m <= 0:
            errors.append("max_aperture_m must be positive")

        if not 0 < self.max_angle_deg <= 90:
            errors.append("max_angle_deg must be between 0 and 90")

        # Check binning
        if self.binning_preset is None and self.binning_table is None:
            errors.append("Either binning_preset or binning_table must be specified")

        # Check processing mode
        if self.processing_mode not in ['sequential', 'parallel']:
            errors.append(f"Invalid processing_mode: {self.processing_mode}")

        return errors

    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0

    def get_output_grid(self) -> OutputGrid:
        """Create OutputGrid from configuration."""
        n_time = int((self.time_max_ms - self.time_min_ms) / self.dt_ms) + 1
        n_inline = (self.inline_max - self.inline_min) // self.inline_step + 1
        n_xline = (self.xline_max - self.xline_min) // self.xline_step + 1

        return OutputGrid(
            n_time=n_time,
            n_inline=n_inline,
            n_xline=n_xline,
            dt_ms=self.dt_ms,
            inline_min=self.inline_min,
            inline_step=self.inline_step,
            xline_min=self.xline_min,
            xline_step=self.xline_step,
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            inline_spacing=self.inline_spacing,
            xline_spacing=self.xline_spacing,
        )

    def get_migration_config(self) -> MigrationConfig:
        """Create MigrationConfig from job configuration."""
        from processors.migration.base import (
            TraveltimeMode,
            WeightMode,
            InterpolationMode,
        )

        return MigrationConfig(
            output_grid=self.get_output_grid(),
            max_aperture_m=self.max_aperture_m,
            max_angle_deg=self.max_angle_deg,
            max_offset_m=self.max_offset_m,
            taper_width_samples=self.taper_width_samples,
            near_offset_taper_m=self.near_offset_taper_m,
            anti_alias_filter=self.anti_alias_filter,
            traveltime_mode=TraveltimeMode.STRAIGHT_RAY,
            weight_mode=WeightMode.NONE,
            interpolation_mode=InterpolationMode.LINEAR,
        )

    def get_velocity_model(self) -> VelocityModel:
        """Create VelocityModel from configuration."""
        if self.velocity_file:
            return VelocityModel.load(self.velocity_file)

        return VelocityModel(
            v0=self.velocity_v0,
            gradient=self.velocity_gradient,
        )

    def get_binning_table(self) -> BinningTable:
        """Get or create binning table."""
        if self.binning_table is not None:
            return self.binning_table

        if self.binning_preset:
            from utils.binning_presets import get_preset
            return get_preset(self.binning_preset)

        raise ValueError("No binning configuration available")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        d = {
            'name': self.name,
            'input_file': self.input_file,
            'output_directory': self.output_directory,
            'velocity_v0': self.velocity_v0,
            'velocity_gradient': self.velocity_gradient,
            'velocity_file': self.velocity_file,
            'time_min_ms': self.time_min_ms,
            'time_max_ms': self.time_max_ms,
            'dt_ms': self.dt_ms,
            'inline_min': self.inline_min,
            'inline_max': self.inline_max,
            'inline_step': self.inline_step,
            'xline_min': self.xline_min,
            'xline_max': self.xline_max,
            'xline_step': self.xline_step,
            'origin_x': self.origin_x,
            'origin_y': self.origin_y,
            'inline_spacing': self.inline_spacing,
            'xline_spacing': self.xline_spacing,
            'max_aperture_m': self.max_aperture_m,
            'max_angle_deg': self.max_angle_deg,
            'max_offset_m': self.max_offset_m,
            'taper_width_samples': self.taper_width_samples,
            'near_offset_taper_m': self.near_offset_taper_m,
            'anti_alias_filter': self.anti_alias_filter,
            'binning_preset': self.binning_preset,
            'header_mapping': self.header_mapping.to_dict(),
            'processing_mode': self.processing_mode,
            'max_workers': self.max_workers,
            'memory_limit_gb': self.memory_limit_gb,
            'enable_checkpointing': self.enable_checkpointing,
            'output_format': self.output_format,
            'create_stack_volume': self.create_stack_volume,
            'create_fold_volume': self.create_fold_volume,
            'compress_output': self.compress_output,
            'created_at': self.created_at,
            'description': self.description,
            'metadata': self.metadata,
        }

        # Serialize binning table if present
        if self.binning_table is not None:
            d['binning_table'] = self.binning_table.to_dict()

        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'MigrationJobConfig':
        """Create from dictionary."""
        # Handle header mapping
        header_mapping_dict = d.pop('header_mapping', {})
        header_mapping = HeaderMapping.from_dict(header_mapping_dict)

        # Handle binning table
        binning_table = None
        if 'binning_table' in d:
            binning_table = BinningTable.from_dict(d.pop('binning_table'))

        return cls(
            header_mapping=header_mapping,
            binning_table=binning_table,
            **{k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        )

    def save(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved migration job config to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'MigrationJobConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            d = json.load(f)
        logger.info(f"Loaded migration job config from {filepath}")
        return cls.from_dict(d)


# =============================================================================
# Job Templates
# =============================================================================

def create_land_3d_template(
    input_file: str,
    output_directory: str,
    velocity_v0: float = 3000.0,
) -> MigrationJobConfig:
    """
    Create template for land 3D survey migration.

    Args:
        input_file: Input SEG-Y file
        output_directory: Output directory
        velocity_v0: Constant velocity (m/s)

    Returns:
        Pre-configured MigrationJobConfig
    """
    return MigrationJobConfig(
        name="Land 3D Migration",
        input_file=input_file,
        output_directory=output_directory,
        velocity_v0=velocity_v0,
        binning_preset='land_3d',
        max_aperture_m=3000.0,
        max_angle_deg=60.0,
        description="Standard land 3D survey migration with 10 offset bins",
    )


def create_marine_template(
    input_file: str,
    output_directory: str,
    velocity_v0: float = 1500.0,
    velocity_gradient: float = 0.3,
) -> MigrationJobConfig:
    """
    Create template for marine survey migration.

    Args:
        input_file: Input SEG-Y file
        output_directory: Output directory
        velocity_v0: Water velocity (m/s)
        velocity_gradient: Velocity gradient (s^-1)

    Returns:
        Pre-configured MigrationJobConfig
    """
    return MigrationJobConfig(
        name="Marine Migration",
        input_file=input_file,
        output_directory=output_directory,
        velocity_v0=velocity_v0,
        velocity_gradient=velocity_gradient,
        binning_preset='marine',
        max_aperture_m=5000.0,
        max_angle_deg=50.0,
        description="Marine streamer migration with 6 offset bins",
    )


def create_wide_azimuth_template(
    input_file: str,
    output_directory: str,
    velocity_v0: float = 2800.0,
) -> MigrationJobConfig:
    """
    Create template for wide-azimuth OVT migration.

    Args:
        input_file: Input SEG-Y file
        output_directory: Output directory
        velocity_v0: Constant velocity (m/s)

    Returns:
        Pre-configured MigrationJobConfig
    """
    return MigrationJobConfig(
        name="Wide Azimuth OVT Migration",
        input_file=input_file,
        output_directory=output_directory,
        velocity_v0=velocity_v0,
        binning_preset='wide_azimuth_ovt',
        max_aperture_m=4000.0,
        max_angle_deg=65.0,
        description="Wide-azimuth migration with 16 OVT bins (4 offset x 4 azimuth)",
    )


def create_full_stack_template(
    input_file: str,
    output_directory: str,
    velocity_v0: float = 2500.0,
) -> MigrationJobConfig:
    """
    Create template for full stack migration.

    Args:
        input_file: Input SEG-Y file
        output_directory: Output directory
        velocity_v0: Constant velocity (m/s)

    Returns:
        Pre-configured MigrationJobConfig
    """
    return MigrationJobConfig(
        name="Full Stack Migration",
        input_file=input_file,
        output_directory=output_directory,
        velocity_v0=velocity_v0,
        binning_preset='full_stack',
        max_aperture_m=5000.0,
        max_angle_deg=70.0,
        create_stack_volume=True,
        description="Single full-stack migration output",
    )


# Template registry
JOB_TEMPLATES = {
    'land_3d': create_land_3d_template,
    'marine': create_marine_template,
    'wide_azimuth_ovt': create_wide_azimuth_template,
    'full_stack': create_full_stack_template,
}


def list_templates() -> List[str]:
    """List available job templates."""
    return list(JOB_TEMPLATES.keys())


def create_from_template(
    template_name: str,
    input_file: str,
    output_directory: str,
    **kwargs
) -> MigrationJobConfig:
    """
    Create job configuration from template.

    Args:
        template_name: Template name
        input_file: Input file path
        output_directory: Output directory
        **kwargs: Additional parameters to override

    Returns:
        MigrationJobConfig
    """
    if template_name not in JOB_TEMPLATES:
        available = ', '.join(JOB_TEMPLATES.keys())
        raise ValueError(f"Unknown template '{template_name}'. Available: {available}")

    config = JOB_TEMPLATES[template_name](input_file, output_directory, **kwargs)
    return config
