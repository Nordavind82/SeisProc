"""
Test Fixtures Package

Synthetic data generators for testing migration algorithms.
"""

from tests.fixtures.synthetic_prestack import (
    create_synthetic_shot_gather,
    create_point_diffractor_data,
    create_dipping_reflector_data,
    create_synthetic_3d_survey,
)

from tests.fixtures.synthetic_diffractor import (
    DiffractorDataset,
    create_diffractor_dataset,
    create_expected_migration_result,
    verify_dataset,
    ricker_wavelet,
)

__all__ = [
    # Pre-stack fixtures
    'create_synthetic_shot_gather',
    'create_point_diffractor_data',
    'create_dipping_reflector_data',
    'create_synthetic_3d_survey',
    # Zero-offset diffractor fixtures
    'DiffractorDataset',
    'create_diffractor_dataset',
    'create_expected_migration_result',
    'verify_dataset',
    'ricker_wavelet',
]
