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
]
