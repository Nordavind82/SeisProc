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

__all__ = [
    'create_synthetic_shot_gather',
    'create_point_diffractor_data',
    'create_dipping_reflector_data',
    'create_synthetic_3d_survey',
]
