"""
Anisotropy Model I/O Utilities

Read/write anisotropy models from various formats:
- Simple text format (z, epsilon, delta)
- JSON format (with metadata)
- NumPy binary format

Supports:
- Constant anisotropy (single epsilon, delta)
- 1D anisotropy functions (epsilon(z), delta(z))
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import json
import logging

from models.anisotropy_model import (
    AnisotropyModel,
    AnisotropyType,
    create_isotropic,
    create_constant_anisotropy,
    create_1d_anisotropy,
)

logger = logging.getLogger(__name__)


def read_anisotropy_text(
    filepath: str,
    z_column: int = 0,
    epsilon_column: int = 1,
    delta_column: int = 2,
    z_unit: str = 's',
    delimiter: Optional[str] = None,
    skip_header: int = 0,
) -> AnisotropyModel:
    """
    Read anisotropy model from text file.

    Supported formats:
    - Three-column: z, epsilon, delta
    - Multi-column: specify column indices
    - Comment lines starting with # are skipped

    Args:
        filepath: Path to anisotropy file
        z_column: Column index for z values (0-based)
        epsilon_column: Column index for epsilon values (0-based)
        delta_column: Column index for delta values (0-based)
        z_unit: 's' (seconds) or 'm' (meters) for z values
        delimiter: Column delimiter (None = whitespace)
        skip_header: Number of header lines to skip

    Returns:
        AnisotropyModel instance
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Anisotropy file not found: {filepath}")

    # Read file
    z_values = []
    epsilon_values = []
    delta_values = []

    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            # Skip header lines
            if i < skip_header:
                continue

            # Skip comments and empty lines
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('//'):
                continue

            # Parse columns
            if delimiter:
                parts = line.split(delimiter)
            else:
                parts = line.split()

            max_col = max(z_column, epsilon_column, delta_column)
            if len(parts) <= max_col:
                logger.warning(f"Skipping line {i+1}: insufficient columns")
                continue

            try:
                z = float(parts[z_column])
                eps = float(parts[epsilon_column])
                delta = float(parts[delta_column])
                z_values.append(z)
                epsilon_values.append(eps)
                delta_values.append(delta)
            except ValueError as e:
                logger.warning(f"Skipping line {i+1}: {e}")
                continue

    if not z_values:
        raise ValueError(f"No valid anisotropy data found in {filepath}")

    z_axis = np.array(z_values, dtype=np.float32)
    epsilon = np.array(epsilon_values, dtype=np.float32)
    delta = np.array(delta_values, dtype=np.float32)

    # Convert z units to seconds if needed
    if z_unit == 'ms':
        z_axis = z_axis / 1000.0

    # Sort by z
    sort_idx = np.argsort(z_axis)
    z_axis = z_axis[sort_idx]
    epsilon = epsilon[sort_idx]
    delta = delta[sort_idx]

    # Create model
    model = create_1d_anisotropy(z_axis, epsilon, delta)
    model.metadata['source_file'] = str(filepath)
    model.metadata['original_z_unit'] = z_unit

    logger.info(
        f"Read anisotropy model from {filepath}: "
        f"{len(z_axis)} samples, z=[{z_axis[0]:.3f}, {z_axis[-1]:.3f}], "
        f"epsilon=[{epsilon.min():.3f}, {epsilon.max():.3f}], "
        f"delta=[{delta.min():.3f}, {delta.max():.3f}]"
    )

    return model


def write_anisotropy_text(
    model: AnisotropyModel,
    filepath: str,
    z_unit: str = 's',
    delimiter: str = '\t',
    header: bool = True,
    precision: int = 4,
) -> None:
    """
    Write anisotropy model to text file.

    Args:
        model: AnisotropyModel instance
        filepath: Output file path
        z_unit: Output z unit ('s' or 'ms')
        delimiter: Column delimiter
        header: Include header line
        precision: Decimal precision for values
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        if header:
            f.write(f"# VTI Anisotropy Model\n")
            f.write(f"# Type: {model.anisotropy_type.value}\n")
            if model.anisotropy_type == AnisotropyType.CONSTANT:
                f.write(f"# Epsilon: {model.epsilon}\n")
                f.write(f"# Delta: {model.delta}\n")
                f.write(f"# Eta: {model.eta}\n")
            f.write(f"# Z({z_unit}){delimiter}Epsilon{delimiter}Delta{delimiter}Eta\n")

        if model.anisotropy_type == AnisotropyType.CONSTANT:
            # Write two lines for constant model
            eta = model.eta
            f.write(f"0.0{delimiter}{model.epsilon:.{precision}f}{delimiter}")
            f.write(f"{model.delta:.{precision}f}{delimiter}{eta:.{precision}f}\n")
            f.write(f"10.0{delimiter}{model.epsilon:.{precision}f}{delimiter}")
            f.write(f"{model.delta:.{precision}f}{delimiter}{eta:.{precision}f}\n")
        else:
            # Write v(z) function
            z_axis = model.z_axis.copy()
            epsilon = model.epsilon
            delta = model.delta

            # Compute eta for each depth
            eta = model.compute_eta(epsilon, delta)

            # Convert z units
            if z_unit == 'ms':
                z_axis = z_axis * 1000.0

            for z, eps, d, e in zip(z_axis, epsilon, delta, eta):
                f.write(f"{z:.{precision}f}{delimiter}")
                f.write(f"{eps:.{precision}f}{delimiter}")
                f.write(f"{d:.{precision}f}{delimiter}")
                f.write(f"{e:.{precision}f}\n")

    logger.info(f"Wrote anisotropy model to {filepath}")


def read_anisotropy_json(filepath: str) -> AnisotropyModel:
    """
    Read anisotropy model from JSON file.

    Args:
        filepath: Path to JSON anisotropy file

    Returns:
        AnisotropyModel instance
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Anisotropy file not found: {filepath}")

    with open(filepath, 'r') as f:
        data = json.load(f)

    model = AnisotropyModel.from_dict(data)
    model.metadata['source_file'] = str(filepath)

    logger.info(f"Read anisotropy model from {filepath}")

    return model


def write_anisotropy_json(
    model: AnisotropyModel,
    filepath: str,
    indent: int = 2,
) -> None:
    """
    Write anisotropy model to JSON file.

    Args:
        model: AnisotropyModel instance
        filepath: Output file path
        indent: JSON indentation
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    data = model.to_dict()

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)

    logger.info(f"Wrote anisotropy model to {filepath}")


def read_anisotropy_npy(filepath: str) -> AnisotropyModel:
    """
    Read anisotropy model from NumPy binary file.

    Expects .npz file with:
    - 'epsilon': epsilon values
    - 'delta': delta values
    - 'z_axis': depth/time axis (optional for constant)
    - 'metadata': JSON string of metadata (optional)

    Args:
        filepath: Path to .npz anisotropy file

    Returns:
        AnisotropyModel instance
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Anisotropy file not found: {filepath}")

    npz = np.load(filepath, allow_pickle=True)

    epsilon = npz['epsilon']
    delta = npz['delta']

    # Handle scalar values
    if epsilon.ndim == 0:
        epsilon = float(epsilon)
    if delta.ndim == 0:
        delta = float(delta)

    z_axis = npz.get('z_axis')
    if z_axis is not None and z_axis.ndim == 0:
        z_axis = None

    metadata = {}
    if 'metadata' in npz:
        try:
            metadata = json.loads(str(npz['metadata']))
        except:
            pass

    metadata['source_file'] = str(filepath)

    if z_axis is not None:
        model = create_1d_anisotropy(z_axis, epsilon, delta)
    elif isinstance(epsilon, (int, float)):
        model = create_constant_anisotropy(epsilon, delta)
    else:
        raise ValueError("Invalid anisotropy data format")

    model.metadata.update(metadata)

    logger.info(f"Read anisotropy model from {filepath}")

    return model


def write_anisotropy_npy(
    model: AnisotropyModel,
    filepath: str,
) -> None:
    """
    Write anisotropy model to NumPy binary file.

    Args:
        model: AnisotropyModel instance
        filepath: Output file path (.npz)
    """
    filepath = Path(filepath)
    if not filepath.suffix:
        filepath = filepath.with_suffix('.npz')
    filepath.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        'epsilon': model.epsilon if isinstance(model.epsilon, np.ndarray) else np.array(model.epsilon),
        'delta': model.delta if isinstance(model.delta, np.ndarray) else np.array(model.delta),
        'metadata': json.dumps(model.metadata),
    }

    if model.z_axis is not None:
        save_dict['z_axis'] = model.z_axis

    np.savez(filepath, **save_dict)

    logger.info(f"Wrote anisotropy model to {filepath}")


def read_anisotropy_auto(filepath: str, **kwargs) -> AnisotropyModel:
    """
    Auto-detect format and read anisotropy model.

    Supported extensions:
    - .txt, .ani, .asc: Text format
    - .json: JSON format
    - .npy, .npz: NumPy format

    Args:
        filepath: Path to anisotropy file
        **kwargs: Additional arguments for specific readers

    Returns:
        AnisotropyModel instance
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix in ['.txt', '.ani', '.asc', '.dat', '.csv']:
        return read_anisotropy_text(filepath, **kwargs)
    elif suffix == '.json':
        return read_anisotropy_json(filepath)
    elif suffix in ['.npy', '.npz']:
        return read_anisotropy_npy(filepath)
    else:
        # Try text format as fallback
        logger.warning(f"Unknown extension {suffix}, trying text format")
        return read_anisotropy_text(filepath, **kwargs)


def write_anisotropy_auto(
    model: AnisotropyModel,
    filepath: str,
    **kwargs,
) -> None:
    """
    Auto-detect format and write anisotropy model.

    Args:
        model: AnisotropyModel instance
        filepath: Output file path
        **kwargs: Additional arguments for specific writers
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix in ['.txt', '.ani', '.asc', '.dat', '.csv']:
        write_anisotropy_text(model, filepath, **kwargs)
    elif suffix == '.json':
        write_anisotropy_json(model, filepath, **kwargs)
    elif suffix in ['.npy', '.npz']:
        write_anisotropy_npy(model, filepath)
    else:
        # Default to JSON
        logger.warning(f"Unknown extension {suffix}, using JSON format")
        write_anisotropy_json(model, filepath, **kwargs)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_anisotropy_from_picks(
    picks: List[Tuple[float, float, float]],
    dz: float = 0.004,
    z_max: Optional[float] = None,
) -> AnisotropyModel:
    """
    Create anisotropy model from z-epsilon-delta picks.

    Args:
        picks: List of (z, epsilon, delta) tuples
        dz: Output z sampling
        z_max: Maximum z (default: max pick z)

    Returns:
        AnisotropyModel interpolated to regular z grid
    """
    if not picks:
        raise ValueError("No anisotropy picks provided")

    # Sort by z
    picks = sorted(picks, key=lambda x: x[0])

    z_picks = np.array([p[0] for p in picks], dtype=np.float32)
    eps_picks = np.array([p[1] for p in picks], dtype=np.float32)
    delta_picks = np.array([p[2] for p in picks], dtype=np.float32)

    if z_max is None:
        z_max = z_picks[-1]

    # Create regular z axis
    z_axis = np.arange(0, z_max + dz, dz, dtype=np.float32)

    # Interpolate
    epsilon = np.interp(z_axis, z_picks, eps_picks).astype(np.float32)
    delta = np.interp(z_axis, z_picks, delta_picks).astype(np.float32)

    model = create_1d_anisotropy(z_axis, epsilon, delta)
    model.metadata['source'] = 'picks'
    model.metadata['n_picks'] = len(picks)

    return model


def estimate_anisotropy_from_wells(
    well_data: Dict[str, np.ndarray],
    method: str = 'average',
) -> AnisotropyModel:
    """
    Estimate anisotropy parameters from well log data.

    Args:
        well_data: Dictionary with keys:
            - 'z': Depth/time array
            - 'vp': P-wave velocity (optional)
            - 'vs': S-wave velocity (optional)
            - 'epsilon': Direct epsilon measurements (optional)
            - 'delta': Direct delta measurements (optional)
        method: Estimation method ('average', 'fit', 'empirical')

    Returns:
        AnisotropyModel estimated from well data
    """
    z_axis = well_data.get('z')
    if z_axis is None:
        raise ValueError("Well data must include 'z' axis")

    # If direct measurements available, use them
    if 'epsilon' in well_data and 'delta' in well_data:
        epsilon = well_data['epsilon']
        delta = well_data['delta']
    else:
        # Use empirical relationships based on lithology
        # This is a simplified example - real estimation would be more complex
        vp = well_data.get('vp')
        vs = well_data.get('vs')

        if vp is not None:
            # Simple empirical estimate for shales
            # epsilon ~ 0.05 - 0.3 depending on clay content
            # delta ~ 0.5 * epsilon typical for shales
            epsilon = np.ones_like(z_axis) * 0.15
            delta = np.ones_like(z_axis) * 0.08
        else:
            raise ValueError(
                "Well data must include either (epsilon, delta) or velocities"
            )

    model = create_1d_anisotropy(z_axis, epsilon, delta)
    model.metadata['source'] = 'well_estimate'
    model.metadata['method'] = method

    return model
