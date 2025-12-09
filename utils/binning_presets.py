"""
Binning Presets for Common Offset Migration Workflows

Provides standard binning configurations for:
- Land 3D surveys (common offset)
- Wide-azimuth surveys (OVT)
- Marine surveys (narrow azimuth)
"""

from typing import List, Tuple, Optional
from models.binning import (
    BinningTable,
    create_common_offset_binning,
    create_uniform_offset_binning,
    create_ovt_binning,
    create_narrow_azimuth_binning,
    create_full_stack_binning,
)


# =============================================================================
# Standard Preset Configurations
# =============================================================================

# Land 3D - 10 offset bins
LAND_3D_OFFSET_RANGES = [
    (0, 200),
    (200, 400),
    (400, 600),
    (600, 800),
    (800, 1000),
    (1000, 1500),
    (1500, 2000),
    (2000, 2500),
    (2500, 3000),
    (3000, 5000),
]

# Marine - 6 offset bins
MARINE_OFFSET_RANGES = [
    (0, 500),
    (500, 1000),
    (1000, 2000),
    (2000, 3000),
    (3000, 4500),
    (4500, 8000),
]

# Wide azimuth - 4 offset bins for OVT
WIDE_AZIMUTH_OFFSET_RANGES = [
    (0, 1000),
    (1000, 2000),
    (2000, 3500),
    (3500, 6000),
]


def get_land_3d_preset() -> BinningTable:
    """
    Standard land 3D survey binning preset.

    10 offset bins from 0 to 5000m with varying widths
    (narrower bins near offset, wider for far offsets).
    Full azimuth coverage.
    """
    table = create_common_offset_binning(
        LAND_3D_OFFSET_RANGES,
        name_prefix="land3d",
    )
    table.name = "Land 3D - 10 Offset Bins"
    return table


def get_marine_preset() -> BinningTable:
    """
    Standard marine survey binning preset.

    6 offset bins for typical marine streamer geometry.
    Full azimuth coverage (though marine is typically narrow azimuth).
    """
    table = create_common_offset_binning(
        MARINE_OFFSET_RANGES,
        name_prefix="marine",
    )
    table.name = "Marine - 6 Offset Bins"
    return table


def get_wide_azimuth_ovt_preset() -> BinningTable:
    """
    Wide-azimuth OVT binning preset.

    4 offset ranges x 4 azimuth sectors = 16 bins.
    Suitable for wide-azimuth land or marine data.
    """
    table = create_ovt_binning(
        WIDE_AZIMUTH_OFFSET_RANGES,
        n_azimuth_sectors=4,
    )
    table.name = "Wide Azimuth OVT - 16 Bins"
    return table


def get_narrow_azimuth_preset(
    inline_azimuth: float = 0.0,
) -> BinningTable:
    """
    Narrow azimuth binning preset.

    4 bins separating inline and crossline directions.

    Args:
        inline_azimuth: Inline direction in degrees from north
    """
    table = create_narrow_azimuth_binning(
        offset_min=0.0,
        offset_max=8000.0,
        inline_azimuth=inline_azimuth,
        azimuth_width=30.0,
    )
    table.name = "Narrow Azimuth - 4 Sectors"
    return table


def get_full_stack_preset(
    offset_max: float = 10000.0,
) -> BinningTable:
    """
    Full stack binning preset.

    Single bin covering all offsets and azimuths.

    Args:
        offset_max: Maximum offset to include
    """
    table = create_full_stack_binning(offset_max)
    table.name = "Full Stack - Single Bin"
    return table


# =============================================================================
# Preset Registry
# =============================================================================

PRESET_REGISTRY = {
    'land_3d': get_land_3d_preset,
    'marine': get_marine_preset,
    'wide_azimuth_ovt': get_wide_azimuth_ovt_preset,
    'narrow_azimuth': get_narrow_azimuth_preset,
    'full_stack': get_full_stack_preset,
}


def get_preset(name: str, **kwargs) -> BinningTable:
    """
    Get a binning preset by name.

    Args:
        name: Preset name (land_3d, marine, wide_azimuth_ovt,
              narrow_azimuth, full_stack)
        **kwargs: Additional arguments passed to preset function

    Returns:
        BinningTable configured for the preset

    Raises:
        ValueError: If preset name is unknown
    """
    if name not in PRESET_REGISTRY:
        available = ', '.join(PRESET_REGISTRY.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")

    return PRESET_REGISTRY[name](**kwargs)


def list_presets() -> List[str]:
    """List available preset names."""
    return list(PRESET_REGISTRY.keys())


def get_preset_description(name: str) -> str:
    """Get description for a preset."""
    descriptions = {
        'land_3d': "10 offset bins for land 3D surveys (0-5000m)",
        'marine': "6 offset bins for marine streamer data (0-8000m)",
        'wide_azimuth_ovt': "16 OVT bins (4 offset x 4 azimuth) for wide-azimuth",
        'narrow_azimuth': "4 bins separating inline/crossline directions",
        'full_stack': "Single bin for full stack migration",
    }
    return descriptions.get(name, "No description available")


# =============================================================================
# Custom Binning Helpers
# =============================================================================

def create_custom_offset_binning(
    offset_min: float,
    offset_max: float,
    n_bins: int,
    logarithmic: bool = False,
) -> BinningTable:
    """
    Create custom offset binning with specified parameters.

    Args:
        offset_min: Minimum offset
        offset_max: Maximum offset
        n_bins: Number of bins
        logarithmic: If True, use logarithmic spacing (denser near offsets)

    Returns:
        BinningTable with custom offset bins
    """
    if logarithmic:
        # Logarithmic spacing (denser at near offsets)
        log_min = max(1, offset_min)  # Avoid log(0)
        log_edges = [
            log_min * (offset_max / log_min) ** (i / n_bins)
            for i in range(n_bins + 1)
        ]
        log_edges[0] = offset_min  # Restore exact min
        ranges = [(log_edges[i], log_edges[i + 1]) for i in range(n_bins)]
    else:
        # Uniform spacing
        bin_width = (offset_max - offset_min) / n_bins
        ranges = [
            (offset_min + i * bin_width, offset_min + (i + 1) * bin_width)
            for i in range(n_bins)
        ]

    return create_common_offset_binning(ranges, name_prefix="custom")


def suggest_binning(
    offsets: 'np.ndarray',
    azimuths: 'np.ndarray',
    target_traces_per_bin: int = 10000,
) -> BinningTable:
    """
    Suggest binning based on data distribution.

    Analyzes input data and suggests appropriate binning.

    Args:
        offsets: Array of trace offsets
        azimuths: Array of trace azimuths
        target_traces_per_bin: Target number of traces per bin

    Returns:
        Suggested BinningTable
    """
    import numpy as np

    n_traces = len(offsets)
    n_bins = max(1, n_traces // target_traces_per_bin)

    # Use quantiles for offset bins to equalize trace counts
    offset_edges = np.percentile(
        offsets,
        np.linspace(0, 100, n_bins + 1)
    )

    ranges = [
        (float(offset_edges[i]), float(offset_edges[i + 1]))
        for i in range(n_bins)
    ]

    table = create_common_offset_binning(ranges, name_prefix="suggested")
    table.name = f"Suggested - {n_bins} Bins"
    table.metadata['source'] = 'auto_suggested'
    table.metadata['target_traces_per_bin'] = target_traces_per_bin

    return table
