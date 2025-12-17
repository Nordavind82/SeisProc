"""
Velocity Model I/O Utilities

Read/write velocity models from various formats:
- Simple text format (t, v pairs)
- CDP-Time-Velocity triplets (spatially varying)
- Inline-Xline-Time-Velocity format (3D)
- SEG-Y velocity files
- JSON format (with metadata)
- NumPy binary format

Supports:
- RMS velocity functions
- Interval velocity functions
- 1D v(z) models
- 2D v(x,z) models
- Constant velocity models
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Union
import numpy as np
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum

from models.velocity_model import (
    VelocityModel,
    VelocityType,
    create_constant_velocity,
    create_linear_gradient_velocity,
    create_from_rms_velocity,
    create_2d_velocity,
    rms_to_interval_velocity,
    interval_to_rms_velocity,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Velocity File Format Detection and Preview
# =============================================================================

class VelocityFileFormat(Enum):
    """Detected velocity file format."""
    ASCII_TV = "ascii_tv"           # Time-Velocity pairs (single location)
    ASCII_CDPTV = "ascii_cdp_tv"    # CDP-Time-Velocity triplets
    ASCII_ILXLTV = "ascii_ilxl_tv"  # Inline-Xline-Time-Velocity
    SEGY = "segy"                   # SEG-Y velocity file
    JSON = "json"                   # JSON format
    NPZ = "npz"                     # NumPy format
    UNKNOWN = "unknown"


@dataclass
class VelocityFileInfo:
    """
    Metadata about a velocity file for preview purposes.

    Attributes:
        path: Path to velocity file
        format: Detected file format
        n_locations: Number of spatial locations (CDPs or inline/xline)
        n_time_samples: Number of time samples per location
        time_range: (min_time, max_time) in seconds
        velocity_range: (min_velocity, max_velocity) in m/s
        cdp_range: (min_cdp, max_cdp) if applicable
        inline_range: (min_inline, max_inline) if applicable
        xline_range: (min_xline, max_xline) if applicable
        sample_interval: Time sample interval if regular
        is_valid: Whether file was successfully parsed
        error_message: Error description if parsing failed
        raw_data: Parsed data arrays (for conversion to VelocityModel)
    """
    path: Path
    format: VelocityFileFormat = VelocityFileFormat.UNKNOWN
    n_locations: int = 0
    n_time_samples: int = 0
    time_range: Tuple[float, float] = (0.0, 0.0)
    velocity_range: Tuple[float, float] = (0.0, 0.0)
    cdp_range: Optional[Tuple[int, int]] = None
    inline_range: Optional[Tuple[int, int]] = None
    xline_range: Optional[Tuple[int, int]] = None
    sample_interval: Optional[float] = None
    is_valid: bool = False
    error_message: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = field(default=None, repr=False)


def detect_velocity_format(filepath: Union[str, Path]) -> VelocityFileFormat:
    """
    Auto-detect velocity file format.

    Args:
        filepath: Path to velocity file

    Returns:
        Detected VelocityFileFormat
    """
    filepath = Path(filepath)

    if not filepath.exists():
        return VelocityFileFormat.UNKNOWN

    suffix = filepath.suffix.lower()

    # Check by extension first
    if suffix in ['.sgy', '.segy']:
        return VelocityFileFormat.SEGY
    elif suffix == '.json':
        return VelocityFileFormat.JSON
    elif suffix in ['.npy', '.npz']:
        return VelocityFileFormat.NPZ

    # For text files, analyze content
    try:
        with open(filepath, 'r') as f:
            lines = []
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('//'):
                    continue
                lines.append(line)
                if len(lines) >= 20:
                    break

        if not lines:
            return VelocityFileFormat.UNKNOWN

        # Analyze column count
        column_counts = []
        for line in lines:
            parts = re.split(r'[,\s\t]+', line)
            numeric_parts = []
            for p in parts:
                try:
                    float(p)
                    numeric_parts.append(p)
                except ValueError:
                    pass
            if numeric_parts:
                column_counts.append(len(numeric_parts))

        if not column_counts:
            return VelocityFileFormat.UNKNOWN

        most_common = max(set(column_counts), key=column_counts.count)

        if most_common == 2:
            return VelocityFileFormat.ASCII_TV
        elif most_common == 3:
            return VelocityFileFormat.ASCII_CDPTV
        elif most_common >= 4:
            return VelocityFileFormat.ASCII_ILXLTV
        else:
            return VelocityFileFormat.UNKNOWN

    except Exception as e:
        logger.warning(f"Error detecting format: {e}")
        return VelocityFileFormat.UNKNOWN


def preview_velocity_file(
    filepath: Union[str, Path],
    inline_byte: Optional[int] = None,
    xline_byte: Optional[int] = None,
) -> VelocityFileInfo:
    """
    Preview velocity file without full loading.

    Args:
        filepath: Path to velocity file
        inline_byte: Custom byte position for inline in SEG-Y (None = use standard 189)
        xline_byte: Custom byte position for xline in SEG-Y (None = use standard 193)

    Returns:
        VelocityFileInfo with metadata and optional raw_data
    """
    filepath = Path(filepath)
    info = VelocityFileInfo(path=filepath)

    if not filepath.exists():
        info.error_message = f"File not found: {filepath}"
        return info

    info.format = detect_velocity_format(filepath)

    try:
        if info.format == VelocityFileFormat.SEGY:
            info = _preview_velocity_segy(filepath, info, inline_byte, xline_byte)
        elif info.format == VelocityFileFormat.JSON:
            info = _preview_velocity_json(filepath, info)
        elif info.format == VelocityFileFormat.NPZ:
            info = _preview_velocity_npz(filepath, info)
        elif info.format == VelocityFileFormat.ASCII_TV:
            info = _preview_velocity_ascii_tv(filepath, info)
        elif info.format == VelocityFileFormat.ASCII_CDPTV:
            info = _preview_velocity_ascii_cdptv(filepath, info)
        elif info.format == VelocityFileFormat.ASCII_ILXLTV:
            info = _preview_velocity_ascii_ilxltv(filepath, info)
        else:
            info.error_message = "Unknown file format"

    except Exception as e:
        info.error_message = f"Preview error: {str(e)}"
        logger.exception(f"Error previewing {filepath}")

    return info


def _preview_velocity_ascii_tv(filepath: Path, info: VelocityFileInfo) -> VelocityFileInfo:
    """Preview Time-Velocity pair format."""
    times = []
    velocities = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            parts = re.split(r'[,\s\t]+', line)
            try:
                if len(parts) >= 2:
                    times.append(float(parts[0]))
                    velocities.append(float(parts[1]))
            except ValueError:
                continue

    if not times:
        info.error_message = "No valid data found"
        return info

    times = np.array(times)
    velocities = np.array(velocities)

    info.n_locations = 1
    info.n_time_samples = len(times)
    info.time_range = (float(times.min()), float(times.max()))
    info.velocity_range = (float(velocities.min()), float(velocities.max()))

    if len(times) > 1:
        dt = np.diff(times)
        if np.allclose(dt, dt[0], rtol=0.01):
            info.sample_interval = float(dt[0])

    info.raw_data = {'times': times, 'velocities': velocities, 'format': 'tv'}
    info.is_valid = True
    return info


def _preview_velocity_ascii_cdptv(filepath: Path, info: VelocityFileInfo) -> VelocityFileInfo:
    """Preview CDP-Time-Velocity triplet format."""
    cdps = []
    times = []
    velocities = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            parts = re.split(r'[,\s\t]+', line)
            try:
                if len(parts) >= 3:
                    cdps.append(int(float(parts[0])))
                    times.append(float(parts[1]))
                    velocities.append(float(parts[2]))
            except ValueError:
                continue

    if not times:
        info.error_message = "No valid data found"
        return info

    cdps = np.array(cdps)
    times = np.array(times)
    velocities = np.array(velocities)
    unique_cdps = np.unique(cdps)

    info.n_locations = len(unique_cdps)
    info.n_time_samples = len(times) // len(unique_cdps) if len(unique_cdps) > 0 else 0
    info.time_range = (float(times.min()), float(times.max()))
    info.velocity_range = (float(velocities.min()), float(velocities.max()))
    info.cdp_range = (int(unique_cdps.min()), int(unique_cdps.max()))

    info.raw_data = {
        'cdps': cdps, 'times': times, 'velocities': velocities,
        'unique_cdps': unique_cdps, 'format': 'cdp_tv'
    }
    info.is_valid = True
    return info


def _preview_velocity_ascii_ilxltv(filepath: Path, info: VelocityFileInfo) -> VelocityFileInfo:
    """Preview Inline-Xline-Time-Velocity format."""
    inlines = []
    xlines = []
    times = []
    velocities = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            parts = re.split(r'[,\s\t]+', line)
            try:
                if len(parts) >= 4:
                    inlines.append(int(float(parts[0])))
                    xlines.append(int(float(parts[1])))
                    times.append(float(parts[2]))
                    velocities.append(float(parts[3]))
            except ValueError:
                continue

    if not times:
        info.error_message = "No valid data found"
        return info

    inlines = np.array(inlines)
    xlines = np.array(xlines)
    times = np.array(times)
    velocities = np.array(velocities)

    unique_inlines = np.unique(inlines)
    unique_xlines = np.unique(xlines)
    n_locations = len(np.unique(list(zip(inlines, xlines))))

    info.n_locations = n_locations
    info.n_time_samples = len(times) // n_locations if n_locations > 0 else 0
    info.time_range = (float(times.min()), float(times.max()))
    info.velocity_range = (float(velocities.min()), float(velocities.max()))
    info.inline_range = (int(unique_inlines.min()), int(unique_inlines.max()))
    info.xline_range = (int(unique_xlines.min()), int(unique_xlines.max()))

    info.raw_data = {
        'inlines': inlines, 'xlines': xlines, 'times': times, 'velocities': velocities,
        'unique_inlines': unique_inlines, 'unique_xlines': unique_xlines, 'format': 'ilxl_tv'
    }
    info.is_valid = True
    return info


def _preview_velocity_segy(
    filepath: Path,
    info: VelocityFileInfo,
    inline_byte: Optional[int] = None,
    xline_byte: Optional[int] = None,
) -> VelocityFileInfo:
    """
    Preview SEG-Y velocity file including inline/xline ranges.

    Args:
        filepath: Path to SEG-Y file
        info: VelocityFileInfo to populate
        inline_byte: Custom byte position for inline (None = use standard INLINE_3D at 189)
        xline_byte: Custom byte position for xline (None = use standard CROSSLINE_3D at 193)

    Returns:
        Updated VelocityFileInfo
    """
    try:
        import segyio

        with segyio.open(str(filepath), 'r', ignore_geometry=True) as f:
            n_traces = f.tracecount
            n_samples = len(f.samples)
            sample_interval = f.bin[segyio.BinField.Interval] / 1e6

            # Determine which byte positions to use for inline/xline
            # Standard positions: INLINE_3D=189, CROSSLINE_3D=193
            # Custom bytes are 1-based, segyio uses enum values
            use_custom_inline = inline_byte is not None
            use_custom_xline = xline_byte is not None

            # Sample first few traces for velocity range
            sample_traces = min(10, n_traces)
            v_min, v_max = float('inf'), float('-inf')
            cdps = []
            inlines = []
            xlines = []

            for i in range(sample_traces):
                trace = f.trace[i]
                v_min = min(v_min, trace.min())
                v_max = max(v_max, trace.max())
                cdps.append(f.header[i][segyio.TraceField.CDP])
                # Use custom byte positions if specified
                if use_custom_inline:
                    inlines.append(f.header[i][inline_byte])
                else:
                    inlines.append(f.header[i][segyio.TraceField.INLINE_3D])
                if use_custom_xline:
                    xlines.append(f.header[i][xline_byte])
                else:
                    xlines.append(f.header[i][segyio.TraceField.CROSSLINE_3D])

            times = f.samples / 1000.0

            info.n_locations = n_traces
            info.n_time_samples = n_samples
            info.time_range = (float(times.min()), float(times.max()))
            info.velocity_range = (float(v_min), float(v_max))
            info.sample_interval = float(sample_interval)

            # Read all headers for full ranges
            all_cdps = []
            all_inlines = []
            all_xlines = []
            for i in range(n_traces):
                all_cdps.append(f.header[i][segyio.TraceField.CDP])
                if use_custom_inline:
                    all_inlines.append(f.header[i][inline_byte])
                else:
                    all_inlines.append(f.header[i][segyio.TraceField.INLINE_3D])
                if use_custom_xline:
                    all_xlines.append(f.header[i][xline_byte])
                else:
                    all_xlines.append(f.header[i][segyio.TraceField.CROSSLINE_3D])

            # Set CDP range if present
            if any(c != 0 for c in all_cdps):
                info.cdp_range = (min(all_cdps), max(all_cdps))

            # Set inline range if present (check for non-zero values)
            if any(il != 0 for il in all_inlines):
                info.inline_range = (min(all_inlines), max(all_inlines))

            # Set xline range if present
            if any(xl != 0 for xl in all_xlines):
                info.xline_range = (min(all_xlines), max(all_xlines))

            info.raw_data = {
                'format': 'segy',
                'filepath': str(filepath),
                'inline_byte': inline_byte,
                'xline_byte': xline_byte,
            }
            info.is_valid = True

    except ImportError:
        info.error_message = "segyio library not installed"
    except Exception as e:
        info.error_message = f"SEG-Y error: {str(e)}"

    return info


def _preview_velocity_json(filepath: Path, info: VelocityFileInfo) -> VelocityFileInfo:
    """Preview JSON velocity file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    vel_type = data.get('velocity_type', 'v_of_z')
    vel_data = data.get('data')

    if isinstance(vel_data, (int, float)):
        info.n_locations = 1
        info.n_time_samples = 1
        info.velocity_range = (float(vel_data), float(vel_data))
    elif isinstance(vel_data, list):
        vel_arr = np.array(vel_data)
        if vel_arr.ndim == 1:
            info.n_locations = 1
            info.n_time_samples = len(vel_arr)
        else:
            info.n_time_samples = vel_arr.shape[0]
            info.n_locations = vel_arr.shape[1] if vel_arr.ndim > 1 else 1
        info.velocity_range = (float(vel_arr.min()), float(vel_arr.max()))

    if 'z_axis' in data:
        z_arr = np.array(data['z_axis'])
        info.time_range = (float(z_arr.min()), float(z_arr.max()))
        if len(z_arr) > 1:
            info.sample_interval = float(z_arr[1] - z_arr[0])

    info.raw_data = {'format': 'json', 'data': data}
    info.is_valid = True
    return info


def _preview_velocity_npz(filepath: Path, info: VelocityFileInfo) -> VelocityFileInfo:
    """Preview NPZ velocity file."""
    npz = np.load(filepath, allow_pickle=True)

    vel_data = npz['data']
    if vel_data.ndim == 0:
        info.n_locations = 1
        info.n_time_samples = 1
        info.velocity_range = (float(vel_data), float(vel_data))
    elif vel_data.ndim == 1:
        info.n_locations = 1
        info.n_time_samples = len(vel_data)
        info.velocity_range = (float(vel_data.min()), float(vel_data.max()))
    else:
        info.n_time_samples = vel_data.shape[0]
        info.n_locations = vel_data.shape[1] if vel_data.ndim > 1 else 1
        info.velocity_range = (float(vel_data.min()), float(vel_data.max()))

    if 'z_axis' in npz:
        z_arr = npz['z_axis']
        if z_arr.ndim > 0:
            info.time_range = (float(z_arr.min()), float(z_arr.max()))
            if len(z_arr) > 1:
                info.sample_interval = float(z_arr[1] - z_arr[0])

    info.raw_data = {'format': 'npz', 'filepath': str(filepath)}
    info.is_valid = True
    return info


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


# =============================================================================
# Extended Velocity File Reading (CDP-TV, SEG-Y)
# =============================================================================

def read_velocity_cdptv(
    filepath: Union[str, Path],
    time_unit: str = 'ms',
    velocity_unit: str = 'm/s',
    velocity_type: str = 'rms',
) -> VelocityModel:
    """
    Read spatially-varying velocity from CDP-Time-Velocity format.

    Format: Three columns (CDP, Time, Velocity) per line.
    Creates a 2D velocity model v(CDP, t).

    Args:
        filepath: Path to ASCII velocity file
        time_unit: Input time unit ('s', 'ms')
        velocity_unit: Input velocity unit ('m/s', 'km/s', 'ft/s')
        velocity_type: 'rms' or 'interval'

    Returns:
        VelocityModel (2D: v(x,z) where x=CDP, z=time)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Velocity file not found: {filepath}")

    # Use preview function to parse the file
    info = preview_velocity_file(filepath)

    if not info.is_valid:
        raise ValueError(f"Failed to read velocity file: {info.error_message}")

    if info.format != VelocityFileFormat.ASCII_CDPTV:
        raise ValueError(f"Expected CDP-TV format, got {info.format.value}")

    raw = info.raw_data
    cdps = raw['cdps']
    times = raw['times']
    velocities = raw['velocities']
    unique_cdps = raw['unique_cdps']

    # Convert units
    if time_unit == 'ms':
        times = times / 1000.0
    if velocity_unit == 'km/s':
        velocities = velocities * 1000.0
    elif velocity_unit == 'ft/s':
        velocities = velocities * 0.3048

    # Build time axis from first CDP
    first_cdp_mask = cdps == unique_cdps[0]
    time_axis = np.sort(times[first_cdp_mask])

    # Build 2D array (n_time x n_cdp)
    n_time = len(time_axis)
    n_cdp = len(unique_cdps)
    vel_2d = np.zeros((n_time, n_cdp), dtype=np.float32)

    for i, cdp in enumerate(unique_cdps):
        mask = cdps == cdp
        cdp_t = times[mask]
        cdp_v = velocities[mask]
        # Sort by time and interpolate to regular grid
        sort_idx = np.argsort(cdp_t)
        vel_2d[:, i] = np.interp(time_axis, cdp_t[sort_idx], cdp_v[sort_idx])

    model = create_2d_velocity(
        data=vel_2d,
        z_axis=time_axis.astype(np.float32),
        x_axis=unique_cdps.astype(np.float32),
        is_time=True,
        metadata={
            'source_file': str(filepath),
            'velocity_type': velocity_type,
            'cdp_range': (int(unique_cdps.min()), int(unique_cdps.max())),
            'original_time_unit': time_unit,
            'original_velocity_unit': velocity_unit,
        }
    )

    logger.info(
        f"Read 2D velocity model from {filepath}: "
        f"{n_cdp} CDPs x {n_time} samples, "
        f"CDP=[{unique_cdps.min()}, {unique_cdps.max()}], "
        f"v=[{vel_2d.min():.0f}, {vel_2d.max():.0f}] m/s"
    )

    return model


def read_velocity_segy(
    filepath: Union[str, Path],
    velocity_type: str = 'rms',
) -> VelocityModel:
    """
    Read velocity model from SEG-Y file.

    Each trace represents velocity function at one CDP/location.
    Trace samples are velocities at regular time intervals.

    Args:
        filepath: Path to SEG-Y velocity file
        velocity_type: 'rms' or 'interval'

    Returns:
        VelocityModel (1D if single trace, 2D if multiple)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Velocity file not found: {filepath}")

    try:
        import segyio
    except ImportError:
        raise ImportError("segyio library required for SEG-Y velocity files")

    with segyio.open(str(filepath), 'r', ignore_geometry=True) as f:
        n_traces = f.tracecount
        n_samples = len(f.samples)
        sample_interval = f.bin[segyio.BinField.Interval] / 1e6  # microsec to sec

        # Build time axis
        times = f.samples / 1000.0  # ms to seconds

        if n_traces == 1:
            # Single trace - 1D model
            velocities = f.trace[0].astype(np.float32)
            model = VelocityModel(
                data=velocities,
                z_axis=times.astype(np.float32),
                is_time=True,
                metadata={
                    'source_file': str(filepath),
                    'velocity_type': velocity_type,
                    'format': 'segy',
                }
            )
        else:
            # Multiple traces - 2D model
            velocities = np.zeros((n_samples, n_traces), dtype=np.float32)
            cdps = np.zeros(n_traces, dtype=np.int32)

            for i in range(n_traces):
                velocities[:, i] = f.trace[i]
                cdps[i] = f.header[i][segyio.TraceField.CDP]

            # Use CDP as x-axis if available
            if np.any(cdps != 0):
                x_axis = cdps.astype(np.float32)
            else:
                x_axis = np.arange(n_traces, dtype=np.float32)

            model = create_2d_velocity(
                data=velocities,
                z_axis=times.astype(np.float32),
                x_axis=x_axis,
                is_time=True,
                metadata={
                    'source_file': str(filepath),
                    'velocity_type': velocity_type,
                    'format': 'segy',
                    'cdp_range': (int(cdps.min()), int(cdps.max())) if np.any(cdps != 0) else None,
                }
            )

    logger.info(f"Read velocity model from SEG-Y {filepath}: {model}")
    return model


def velocity_info_to_model(
    info: VelocityFileInfo,
    velocity_type: str = 'rms',
    time_unit: str = 'ms',
    velocity_unit: str = 'm/s',
) -> VelocityModel:
    """
    Convert VelocityFileInfo (from preview) to VelocityModel.

    Args:
        info: VelocityFileInfo from preview_velocity_file()
        velocity_type: 'rms' or 'interval'
        time_unit: Time unit for ASCII files ('s' or 'ms')
        velocity_unit: Velocity unit for ASCII files ('m/s', 'km/s', 'ft/s')

    Returns:
        VelocityModel instance

    Raises:
        ValueError: If info is invalid
    """
    if not info.is_valid:
        raise ValueError(f"Invalid velocity file: {info.error_message}")

    raw = info.raw_data
    fmt = raw.get('format', '')

    if fmt == 'tv':
        # Single location - 1D model
        times = raw['times'].copy()
        velocities = raw['velocities'].copy()

        # Convert units
        if time_unit == 'ms':
            times = times / 1000.0
        if velocity_unit == 'km/s':
            velocities = velocities * 1000.0
        elif velocity_unit == 'ft/s':
            velocities = velocities * 0.3048

        return VelocityModel(
            data=velocities.astype(np.float32),
            z_axis=times.astype(np.float32),
            is_time=True,
            metadata={
                'source_file': str(info.path),
                'velocity_type': velocity_type,
            }
        )

    elif fmt == 'cdp_tv':
        # Use the dedicated reader
        return read_velocity_cdptv(
            info.path,
            time_unit=time_unit,
            velocity_unit=velocity_unit,
            velocity_type=velocity_type,
        )

    elif fmt == 'ilxl_tv':
        # Inline-Xline format - create 2D model using inline as x-axis
        inlines = raw['inlines']
        times = raw['times'].copy()
        velocities = raw['velocities'].copy()
        unique_inlines = raw['unique_inlines']

        # Convert units
        if time_unit == 'ms':
            times = times / 1000.0
        if velocity_unit == 'km/s':
            velocities = velocities * 1000.0
        elif velocity_unit == 'ft/s':
            velocities = velocities * 0.3048

        # Build time axis from first inline
        first_il_mask = inlines == unique_inlines[0]
        time_axis = np.sort(times[first_il_mask])

        # Build 2D array
        n_time = len(time_axis)
        n_il = len(unique_inlines)
        vel_2d = np.zeros((n_time, n_il), dtype=np.float32)

        for i, il in enumerate(unique_inlines):
            mask = inlines == il
            il_t = times[mask]
            il_v = velocities[mask]
            sort_idx = np.argsort(il_t)
            vel_2d[:, i] = np.interp(time_axis, il_t[sort_idx], il_v[sort_idx])

        return create_2d_velocity(
            data=vel_2d,
            z_axis=time_axis.astype(np.float32),
            x_axis=unique_inlines.astype(np.float32),
            is_time=True,
            metadata={
                'source_file': str(info.path),
                'velocity_type': velocity_type,
                'inline_range': info.inline_range,
                'xline_range': info.xline_range,
            }
        )

    elif fmt == 'segy':
        return read_velocity_segy(info.path, velocity_type=velocity_type)

    elif fmt == 'json':
        return read_velocity_json(str(info.path))

    elif fmt == 'npz':
        return read_velocity_npy(str(info.path))

    else:
        raise ValueError(f"Unknown velocity format: {fmt}")


def get_velocity_at_cdp(
    model: VelocityModel,
    cdp: int,
    times: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract velocity function at a specific CDP location.

    For 2D models, interpolates to the requested CDP.
    For 1D models, returns the single velocity function.

    Args:
        model: VelocityModel (1D or 2D)
        cdp: CDP number to extract
        times: Optional time values to interpolate to

    Returns:
        Tuple of (times, velocities) arrays
    """
    if model.velocity_type == VelocityType.CONSTANT:
        if times is None:
            times = np.array([0.0, 6.0], dtype=np.float32)
        return times, np.full_like(times, model.data)

    elif model.velocity_type == VelocityType.V_OF_Z:
        if times is None:
            return model.z_axis, model.data
        else:
            return times, np.interp(times, model.z_axis, model.data).astype(np.float32)

    elif model.velocity_type == VelocityType.V_OF_XZ:
        if times is None:
            times = model.z_axis

        # Interpolate velocity at CDP location
        velocities = model.get_velocity_at(times, x=float(cdp))
        return times, velocities

    else:
        raise ValueError(f"Unsupported velocity type: {model.velocity_type}")


def get_velocity_summary(model: VelocityModel) -> str:
    """
    Generate human-readable summary of velocity model.

    Args:
        model: VelocityModel instance

    Returns:
        Formatted summary string
    """
    lines = [f"Velocity Model: {model.velocity_type.value}"]

    if model.velocity_type == VelocityType.CONSTANT:
        lines.append(f"  Constant: {model.data:.0f} m/s")
    elif model.velocity_type == VelocityType.V_OF_Z:
        lines.extend([
            f"  Samples: {len(model.data)}",
            f"  Time range: {model.z_axis[0]:.3f} - {model.z_axis[-1]:.3f} s",
            f"  Velocity range: {model.data.min():.0f} - {model.data.max():.0f} m/s",
            f"  V0: {model.v0:.0f} m/s",
            f"  Gradient: {model.gradient:.2f} m/s/s" if model.gradient else "",
        ])
    elif model.velocity_type == VelocityType.V_OF_XZ:
        lines.extend([
            f"  Shape: {model.data.shape[1]} locations x {model.data.shape[0]} samples",
            f"  Time range: {model.z_axis[0]:.3f} - {model.z_axis[-1]:.3f} s",
            f"  X range: {model.x_axis[0]:.0f} - {model.x_axis[-1]:.0f}",
            f"  Velocity range: {model.data.min():.0f} - {model.data.max():.0f} m/s",
        ])

    if model.metadata.get('velocity_type'):
        lines.append(f"  Type: {model.metadata['velocity_type']}")
    if model.metadata.get('source_file'):
        lines.append(f"  Source: {Path(model.metadata['source_file']).name}")

    return "\n".join(filter(None, lines))
