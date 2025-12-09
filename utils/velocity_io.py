"""
Velocity Model I/O Utilities

Read/write velocity models from various formats:
- Simple text format (t, v pairs)
- JSON format (with metadata)
- NumPy binary format

Supports:
- RMS velocity functions
- Interval velocity functions
- 1D v(z) models
- Constant velocity models
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import json
import logging

from models.velocity_model import (
    VelocityModel,
    VelocityType,
    create_constant_velocity,
    create_linear_gradient_velocity,
    create_from_rms_velocity,
    rms_to_interval_velocity,
    interval_to_rms_velocity,
)

logger = logging.getLogger(__name__)


def read_velocity_text(
    filepath: str,
    velocity_type: str = 'rms',
    time_column: int = 0,
    velocity_column: int = 1,
    time_unit: str = 'ms',
    velocity_unit: str = 'm/s',
    delimiter: Optional[str] = None,
    skip_header: int = 0,
) -> VelocityModel:
    """
    Read velocity model from text file.

    Supported formats:
    - Two-column: time, velocity
    - Multi-column: specify time_column and velocity_column
    - Comment lines starting with # are skipped

    Args:
        filepath: Path to velocity file
        velocity_type: 'rms', 'interval', or 'instantaneous'
        time_column: Column index for time values (0-based)
        velocity_column: Column index for velocity values (0-based)
        time_unit: 'ms' or 's' for time values
        velocity_unit: 'm/s', 'ft/s', 'km/s'
        delimiter: Column delimiter (None = whitespace)
        skip_header: Number of header lines to skip

    Returns:
        VelocityModel instance
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Velocity file not found: {filepath}")

    # Read file
    times = []
    velocities = []

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

            if len(parts) <= max(time_column, velocity_column):
                logger.warning(f"Skipping line {i+1}: insufficient columns")
                continue

            try:
                t = float(parts[time_column])
                v = float(parts[velocity_column])
                times.append(t)
                velocities.append(v)
            except ValueError as e:
                logger.warning(f"Skipping line {i+1}: {e}")
                continue

    if not times:
        raise ValueError(f"No valid velocity data found in {filepath}")

    times = np.array(times, dtype=np.float32)
    velocities = np.array(velocities, dtype=np.float32)

    # Convert time units to seconds
    if time_unit == 'ms':
        times = times / 1000.0
    elif time_unit == 'us':
        times = times / 1000000.0

    # Convert velocity units to m/s
    if velocity_unit == 'ft/s':
        velocities = velocities * 0.3048
    elif velocity_unit == 'km/s':
        velocities = velocities * 1000.0
    elif velocity_unit == 'ft/ms':
        velocities = velocities * 304.8

    # Sort by time
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    velocities = velocities[sort_idx]

    # Create model
    model = VelocityModel(
        data=velocities,
        z_axis=times,
        is_time=True,
        metadata={
            'source_file': str(filepath),
            'velocity_type': velocity_type,
            'original_time_unit': time_unit,
            'original_velocity_unit': velocity_unit,
        }
    )

    logger.info(
        f"Read velocity model from {filepath}: "
        f"{len(times)} samples, t=[{times[0]:.3f}, {times[-1]:.3f}]s, "
        f"v=[{velocities.min():.0f}, {velocities.max():.0f}] m/s"
    )

    return model


def write_velocity_text(
    model: VelocityModel,
    filepath: str,
    time_unit: str = 'ms',
    velocity_unit: str = 'm/s',
    delimiter: str = '\t',
    header: bool = True,
    precision: int = 2,
) -> None:
    """
    Write velocity model to text file.

    Args:
        model: VelocityModel instance
        filepath: Output file path
        time_unit: Output time unit ('ms' or 's')
        velocity_unit: Output velocity unit ('m/s', 'ft/s', 'km/s')
        delimiter: Column delimiter
        header: Include header line
        precision: Decimal precision for values
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if model.velocity_type == VelocityType.CONSTANT:
        # Write single constant value
        with open(filepath, 'w') as f:
            if header:
                f.write(f"# Constant velocity model\n")
                f.write(f"# Time({time_unit}){delimiter}Velocity({velocity_unit})\n")

            v = model.data
            if velocity_unit == 'ft/s':
                v = v / 0.3048
            elif velocity_unit == 'km/s':
                v = v / 1000.0

            f.write(f"0.0{delimiter}{v:.{precision}f}\n")
            f.write(f"10000.0{delimiter}{v:.{precision}f}\n")
    else:
        # Write v(z) function
        times = model.z_axis.copy()
        velocities = model.data.copy()

        # Convert units
        if time_unit == 'ms':
            times = times * 1000.0

        if velocity_unit == 'ft/s':
            velocities = velocities / 0.3048
        elif velocity_unit == 'km/s':
            velocities = velocities / 1000.0

        with open(filepath, 'w') as f:
            if header:
                f.write(f"# Velocity model: {model.velocity_type.value}\n")
                f.write(f"# v0={model.v0:.0f} m/s, gradient={model.gradient:.4f}\n")
                f.write(f"# Time({time_unit}){delimiter}Velocity({velocity_unit})\n")

            for t, v in zip(times, velocities):
                f.write(f"{t:.{precision}f}{delimiter}{v:.{precision}f}\n")

    logger.info(f"Wrote velocity model to {filepath}")


def read_velocity_json(filepath: str) -> VelocityModel:
    """
    Read velocity model from JSON file.

    Args:
        filepath: Path to JSON velocity file

    Returns:
        VelocityModel instance
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Velocity file not found: {filepath}")

    with open(filepath, 'r') as f:
        data = json.load(f)

    model = VelocityModel.from_dict(data)
    model.metadata['source_file'] = str(filepath)

    logger.info(f"Read velocity model from {filepath}: {model}")

    return model


def write_velocity_json(
    model: VelocityModel,
    filepath: str,
    indent: int = 2,
) -> None:
    """
    Write velocity model to JSON file.

    Args:
        model: VelocityModel instance
        filepath: Output file path
        indent: JSON indentation
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    data = model.to_dict()

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)

    logger.info(f"Wrote velocity model to {filepath}")


def read_velocity_npy(filepath: str) -> VelocityModel:
    """
    Read velocity model from NumPy binary file.

    Expects .npz file with:
    - 'data': velocity values
    - 'z_axis': depth/time axis (optional for constant)
    - 'metadata': JSON string of metadata (optional)

    Args:
        filepath: Path to .npz velocity file

    Returns:
        VelocityModel instance
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Velocity file not found: {filepath}")

    npz = np.load(filepath, allow_pickle=True)

    data = npz['data']
    if data.ndim == 0:
        # Scalar
        data = float(data)

    z_axis = npz.get('z_axis')
    if z_axis is not None and z_axis.ndim == 0:
        z_axis = None

    x_axis = npz.get('x_axis')
    y_axis = npz.get('y_axis')

    metadata = {}
    if 'metadata' in npz:
        try:
            metadata = json.loads(str(npz['metadata']))
        except:
            pass

    metadata['source_file'] = str(filepath)

    is_time = npz.get('is_time', True)
    if isinstance(is_time, np.ndarray):
        is_time = bool(is_time)

    model = VelocityModel(
        data=data,
        z_axis=z_axis,
        x_axis=x_axis,
        y_axis=y_axis,
        is_time=is_time,
        metadata=metadata,
    )

    logger.info(f"Read velocity model from {filepath}: {model}")

    return model


def write_velocity_npy(
    model: VelocityModel,
    filepath: str,
) -> None:
    """
    Write velocity model to NumPy binary file.

    Args:
        model: VelocityModel instance
        filepath: Output file path (.npz)
    """
    filepath = Path(filepath)
    if not filepath.suffix:
        filepath = filepath.with_suffix('.npz')
    filepath.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        'data': model.data if isinstance(model.data, np.ndarray) else np.array(model.data),
        'is_time': model.is_time,
        'metadata': json.dumps(model.metadata),
    }

    if model.z_axis is not None:
        save_dict['z_axis'] = model.z_axis
    if model.x_axis is not None:
        save_dict['x_axis'] = model.x_axis
    if model.y_axis is not None:
        save_dict['y_axis'] = model.y_axis

    np.savez(filepath, **save_dict)

    logger.info(f"Wrote velocity model to {filepath}")


def read_velocity_auto(filepath: str, **kwargs) -> VelocityModel:
    """
    Auto-detect format and read velocity model.

    Supported extensions:
    - .txt, .vel, .asc: Text format
    - .json: JSON format
    - .npy, .npz: NumPy format

    Args:
        filepath: Path to velocity file
        **kwargs: Additional arguments for specific readers

    Returns:
        VelocityModel instance
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix in ['.txt', '.vel', '.asc', '.dat', '.csv']:
        return read_velocity_text(filepath, **kwargs)
    elif suffix == '.json':
        return read_velocity_json(filepath)
    elif suffix in ['.npy', '.npz']:
        return read_velocity_npy(filepath)
    else:
        # Try text format as fallback
        logger.warning(f"Unknown extension {suffix}, trying text format")
        return read_velocity_text(filepath, **kwargs)


def write_velocity_auto(
    model: VelocityModel,
    filepath: str,
    **kwargs,
) -> None:
    """
    Auto-detect format and write velocity model.

    Args:
        model: VelocityModel instance
        filepath: Output file path
        **kwargs: Additional arguments for specific writers
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix in ['.txt', '.vel', '.asc', '.dat', '.csv']:
        write_velocity_text(model, filepath, **kwargs)
    elif suffix == '.json':
        write_velocity_json(model, filepath, **kwargs)
    elif suffix in ['.npy', '.npz']:
        write_velocity_npy(model, filepath)
    else:
        # Default to JSON
        logger.warning(f"Unknown extension {suffix}, using JSON format")
        write_velocity_json(model, filepath, **kwargs)


# =============================================================================
# Velocity Conversion Utilities
# =============================================================================

def convert_velocity_file(
    input_path: str,
    output_path: str,
    input_type: str = 'rms',
    output_type: str = 'interval',
    **read_kwargs,
) -> VelocityModel:
    """
    Read velocity file, convert type, and write to new file.

    Args:
        input_path: Input velocity file
        output_path: Output velocity file
        input_type: Input velocity type ('rms' or 'interval')
        output_type: Output velocity type ('rms' or 'interval')
        **read_kwargs: Additional arguments for reader

    Returns:
        Converted VelocityModel
    """
    # Read input
    model = read_velocity_auto(input_path, velocity_type=input_type, **read_kwargs)

    if model.velocity_type != VelocityType.V_OF_Z:
        raise ValueError("Velocity conversion only supported for 1D v(z) models")

    # Convert if needed
    if input_type != output_type:
        if input_type == 'rms' and output_type == 'interval':
            t_axis, v_int = rms_to_interval_velocity(model.z_axis, model.data)
            model = VelocityModel(
                data=v_int,
                z_axis=t_axis,
                is_time=model.is_time,
                metadata={**model.metadata, 'velocity_type': 'interval'},
            )
        elif input_type == 'interval' and output_type == 'rms':
            t_axis, v_rms = interval_to_rms_velocity(model.z_axis, model.data)
            model = VelocityModel(
                data=v_rms,
                z_axis=t_axis,
                is_time=model.is_time,
                metadata={**model.metadata, 'velocity_type': 'rms'},
            )

    # Write output
    write_velocity_auto(model, output_path)

    return model


def create_velocity_from_picks(
    picks: List[Tuple[float, float]],
    dt: float = 0.004,
    t_max: Optional[float] = None,
    velocity_type: str = 'rms',
) -> VelocityModel:
    """
    Create velocity model from time-velocity picks.

    Args:
        picks: List of (time_s, velocity_m/s) tuples
        dt: Output time sampling (seconds)
        t_max: Maximum time (default: max pick time)
        velocity_type: 'rms' or 'interval'

    Returns:
        VelocityModel interpolated to regular time grid
    """
    if not picks:
        raise ValueError("No velocity picks provided")

    # Sort by time
    picks = sorted(picks, key=lambda x: x[0])

    t_picks = np.array([p[0] for p in picks], dtype=np.float32)
    v_picks = np.array([p[1] for p in picks], dtype=np.float32)

    if t_max is None:
        t_max = t_picks[-1]

    # Create regular time axis
    t_axis = np.arange(0, t_max + dt, dt, dtype=np.float32)

    # Interpolate velocities
    v_interp = np.interp(t_axis, t_picks, v_picks).astype(np.float32)

    model = VelocityModel(
        data=v_interp,
        z_axis=t_axis,
        is_time=True,
        metadata={
            'velocity_type': velocity_type,
            'source': 'picks',
            'n_picks': len(picks),
        }
    )

    return model
