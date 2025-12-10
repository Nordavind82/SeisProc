"""
Migration Processors Package

GPU-accelerated Kirchhoff Pre-Stack Time Migration components.
"""

from processors.migration.base_migrator import BaseMigrator, MigrationResult
from processors.migration.traveltime import (
    TraveltimeCalculator,
    StraightRayTraveltime,
    CurvedRayTraveltime,
    get_traveltime_calculator,
)
from processors.migration.weights import (
    AmplitudeWeight,
    StandardWeight,
    TrueAmplitudeWeight,
    get_amplitude_weight,
)
from processors.migration.interpolation import (
    TraceInterpolator,
    interpolate_batch,
    interpolate_at_traveltimes,
)
from processors.migration.aperture import (
    ApertureController,
    compute_aperture_indices,
)
from processors.migration.kirchhoff_migrator import (
    KirchhoffMigrator,
    create_kirchhoff_migrator,
)
from processors.migration.antialias import (
    AntialiasFilter,
    AntialiasMethod,
    DipEstimator,
    get_antialias_filter,
)
from processors.migration.traveltime_cache import (
    TraveltimeTable,
    TraveltimeCache,
    TraveltimeTableBuilder,
    CachedTraveltimeCalculator,
    create_traveltime_cache,
    create_cached_calculator,
)
from processors.migration.checkpoint import (
    CheckpointManager,
    JobCheckpoint,
    BinCheckpoint,
    BinStatus,
    IntermediateVolumeSaver,
    create_checkpoint_manager,
    resume_from_checkpoint,
    find_resumable_jobs,
)

# New high-performance engine (PSTM Redesign)
from processors.migration.geometry_preprocessor import (
    GeometryPreprocessor,
    PrecomputedGeometry,
    compute_output_indices,
    compute_traveltimes,
    compute_weights,
)
from processors.migration.kirchhoff_kernel import (
    KirchhoffKernel,
    interpolate_traces,
    scatter_add_migration,
    normalize_by_fold,
)
from processors.migration.migration_engine import MigrationEngine
from processors.migration.config_adapter import (
    ConfigAdapter,
    MigrationParams,
    create_adapter_from_wizard,
)

__all__ = [
    # Base classes
    'BaseMigrator',
    'MigrationResult',
    # Traveltime
    'TraveltimeCalculator',
    'StraightRayTraveltime',
    'CurvedRayTraveltime',
    'get_traveltime_calculator',
    # Weights
    'AmplitudeWeight',
    'StandardWeight',
    'TrueAmplitudeWeight',
    'get_amplitude_weight',
    # Interpolation
    'TraceInterpolator',
    'interpolate_batch',
    'interpolate_at_traveltimes',
    # Aperture
    'ApertureController',
    'compute_aperture_indices',
    # Main migrator
    'KirchhoffMigrator',
    'create_kirchhoff_migrator',
    # Antialiasing
    'AntialiasFilter',
    'AntialiasMethod',
    'DipEstimator',
    'get_antialias_filter',
    # Traveltime caching
    'TraveltimeTable',
    'TraveltimeCache',
    'TraveltimeTableBuilder',
    'CachedTraveltimeCalculator',
    'create_traveltime_cache',
    'create_cached_calculator',
    # Checkpointing
    'CheckpointManager',
    'JobCheckpoint',
    'BinCheckpoint',
    'BinStatus',
    'IntermediateVolumeSaver',
    'create_checkpoint_manager',
    'resume_from_checkpoint',
    'find_resumable_jobs',
    # New high-performance engine
    'MigrationEngine',
    'GeometryPreprocessor',
    'PrecomputedGeometry',
    'KirchhoffKernel',
    'compute_output_indices',
    'compute_traveltimes',
    'compute_weights',
    'interpolate_traces',
    'scatter_add_migration',
    'normalize_by_fold',
    # Config adapter
    'ConfigAdapter',
    'MigrationParams',
    'create_adapter_from_wizard',
]
