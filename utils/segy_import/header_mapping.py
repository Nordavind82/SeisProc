"""
Header mapping configuration for SEG-Y import.
Manages standard and custom trace header mappings.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import struct
import json
from .computed_headers import ComputedHeaderField, ComputedHeaderProcessor


@dataclass
class HeaderField:
    """
    Definition of a trace header field.

    Attributes:
        name: Field name (e.g., 'cdp', 'offset', 'inline')
        byte_position: Starting byte position (1-based, SEG-Y convention)
        format: Struct format ('i'=int32, 'h'=int16, 'f'=float32, etc.)
        description: Human-readable description
    """
    name: str
    byte_position: int
    format: str
    description: str = ""

    @property
    def byte_size(self) -> int:
        """Get size in bytes for this format."""
        return struct.calcsize(self.format)

    def read_value(self, header_bytes: bytes) -> any:
        """
        Read value from header bytes.

        Args:
            header_bytes: 240-byte trace header

        Returns:
            Unpacked value
        """
        if self.byte_position < 1 or self.byte_position > 240:
            raise ValueError(f"Invalid byte position: {self.byte_position}")

        # Convert to 0-based index
        start = self.byte_position - 1
        end = start + self.byte_size

        # Extract bytes and unpack (big-endian)
        value_bytes = header_bytes[start:end]
        return struct.unpack('>' + self.format, value_bytes)[0]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'byte_position': self.byte_position,
            'format': self.format,
            'description': self.description
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'HeaderField':
        """Create from dictionary."""
        return cls(**data)


class StandardHeaders:
    """
    Standard SEG-Y Rev 1 trace header definitions.
    Following SEG-Y specification byte positions.
    """

    TRACE_SEQUENCE_LINE = HeaderField('trace_sequence_line', 1, 'i', 'Trace sequence number within line')
    TRACE_SEQUENCE_FILE = HeaderField('trace_sequence_file', 5, 'i', 'Trace sequence number within file')
    FIELD_RECORD = HeaderField('field_record', 9, 'i', 'Original field record number')
    TRACE_NUMBER = HeaderField('trace_number', 13, 'i', 'Trace number within field record')

    ENERGY_SOURCE_POINT = HeaderField('energy_source_point', 17, 'i', 'Energy source point number')
    CDP = HeaderField('cdp', 21, 'i', 'CDP ensemble number')
    TRACE_NUMBER_CDP = HeaderField('trace_number_cdp', 25, 'i', 'Trace number within CDP')

    TRACE_IDENTIFICATION_CODE = HeaderField('trace_id_code', 29, 'h', 'Trace identification code')

    # Offsets and distances
    OFFSET = HeaderField('offset', 37, 'i', 'Distance from source to receiver')
    RECEIVER_GROUP_ELEVATION = HeaderField('receiver_elevation', 41, 'i', 'Receiver group elevation')
    SURFACE_ELEVATION_SOURCE = HeaderField('source_elevation', 45, 'i', 'Surface elevation at source')
    SOURCE_DEPTH = HeaderField('source_depth', 49, 'i', 'Source depth below surface')

    # Coordinates (scalable)
    SCALAR_COORDINATES = HeaderField('scalar_coord', 71, 'h', 'Scalar for coordinates')
    SOURCE_X = HeaderField('source_x', 73, 'i', 'Source coordinate X')
    SOURCE_Y = HeaderField('source_y', 77, 'i', 'Source coordinate Y')
    RECEIVER_X = HeaderField('receiver_x', 81, 'i', 'Receiver coordinate X')
    RECEIVER_Y = HeaderField('receiver_y', 85, 'i', 'Receiver coordinate Y')

    # Time and samples
    SAMPLE_COUNT = HeaderField('sample_count', 115, 'h', 'Number of samples in trace')
    SAMPLE_INTERVAL = HeaderField('sample_interval', 117, 'h', 'Sample interval (microseconds)')

    # 3D specific
    INLINE = HeaderField('inline', 189, 'i', 'Inline number (3D)')
    CROSSLINE = HeaderField('crossline', 193, 'i', 'Crossline number (3D)')

    @classmethod
    def get_all_standard(cls) -> List[HeaderField]:
        """Get all standard header fields."""
        return [
            cls.TRACE_SEQUENCE_LINE,
            cls.TRACE_SEQUENCE_FILE,
            cls.FIELD_RECORD,
            cls.TRACE_NUMBER,
            cls.ENERGY_SOURCE_POINT,
            cls.CDP,
            cls.TRACE_NUMBER_CDP,
            cls.TRACE_IDENTIFICATION_CODE,
            cls.OFFSET,
            cls.RECEIVER_GROUP_ELEVATION,
            cls.SURFACE_ELEVATION_SOURCE,
            cls.SOURCE_DEPTH,
            cls.SCALAR_COORDINATES,
            cls.SOURCE_X,
            cls.SOURCE_Y,
            cls.RECEIVER_X,
            cls.RECEIVER_Y,
            cls.SAMPLE_COUNT,
            cls.SAMPLE_INTERVAL,
            cls.INLINE,
            cls.CROSSLINE,
        ]

    @classmethod
    def get_minimal(cls) -> List[HeaderField]:
        """Get minimal set of headers for basic processing."""
        return [
            cls.TRACE_SEQUENCE_FILE,
            cls.CDP,
            cls.OFFSET,
            cls.SAMPLE_COUNT,
            cls.SAMPLE_INTERVAL,
        ]


class HeaderMapping:
    """
    Manages trace header mapping configuration.
    Includes standard, custom, and computed header definitions.
    """

    def __init__(self):
        self.fields: Dict[str, HeaderField] = {}
        self.ensemble_keys: List[str] = []  # Headers that define ensemble boundaries
        self.computed_fields: List[ComputedHeaderField] = []  # Computed headers
        self._computed_processor: Optional[ComputedHeaderProcessor] = None

    def add_field(self, field: HeaderField):
        """Add a header field to the mapping."""
        self.fields[field.name] = field

    def remove_field(self, name: str):
        """Remove a header field from the mapping."""
        if name in self.fields:
            del self.fields[name]
            # Remove from ensemble keys if present
            if name in self.ensemble_keys:
                self.ensemble_keys.remove(name)

    def set_ensemble_keys(self, keys: List[str]):
        """
        Set which headers define ensemble boundaries.

        Args:
            keys: List of header field names that define ensembles
                 (e.g., ['cdp'] for CDP gathers, ['inline', 'crossline'] for shots)
        """
        # Validate all keys exist
        for key in keys:
            if key not in self.fields:
                raise ValueError(f"Ensemble key '{key}' not in header mapping")
        self.ensemble_keys = keys

    def add_standard_headers(self, fields: Optional[List[HeaderField]] = None):
        """
        Add standard SEG-Y headers.

        Args:
            fields: List of standard headers to add (default: all standard)
        """
        if fields is None:
            fields = StandardHeaders.get_all_standard()

        for field in fields:
            self.add_field(field)

    def add_computed_field(self, field: ComputedHeaderField):
        """
        Add a computed header field.

        Args:
            field: Computed header field definition
        """
        # Check for name conflicts with raw headers
        if field.name in self.fields:
            raise ValueError(f"Computed header name '{field.name}' conflicts with existing raw header")

        # Check for duplicate computed headers
        if any(f.name == field.name for f in self.computed_fields):
            raise ValueError(f"Computed header '{field.name}' already exists")

        self.computed_fields.append(field)
        self._computed_processor = None  # Invalidate processor cache

    def remove_computed_field(self, name: str):
        """
        Remove a computed header field.

        Args:
            name: Name of the computed field to remove
        """
        self.computed_fields = [f for f in self.computed_fields if f.name != name]
        self._computed_processor = None  # Invalidate processor cache

    def get_computed_processor(self) -> Optional[ComputedHeaderProcessor]:
        """
        Get or create the computed header processor.

        Returns:
            ComputedHeaderProcessor if computed fields exist, None otherwise
        """
        if not self.computed_fields:
            return None

        if self._computed_processor is None:
            self._computed_processor = ComputedHeaderProcessor(self.computed_fields)

        return self._computed_processor

    def has_computed_headers(self) -> bool:
        """Check if any computed headers are configured."""
        return len(self.computed_fields) > 0

    def read_headers(self, header_bytes: bytes, trace_idx: int = -1,
                    compute_headers: bool = True) -> Dict[str, any]:
        """
        Read all configured headers from trace header bytes.
        Optionally computes derived headers.

        Args:
            header_bytes: 240-byte trace header
            trace_idx: Trace index (for computed header error reporting)
            compute_headers: Whether to compute derived headers (default: True)

        Returns:
            Dictionary of header name -> value (includes computed headers if enabled)
        """
        # Read raw headers
        headers = {}
        for name, field in self.fields.items():
            try:
                headers[name] = field.read_value(header_bytes)
            except Exception as e:
                headers[name] = None
                print(f"Warning: Failed to read header '{name}': {e}")

        # Compute derived headers if enabled
        if compute_headers and self.has_computed_headers():
            processor = self.get_computed_processor()
            if processor:
                computed = processor.compute_headers(headers, trace_idx)
                headers.update(computed)

        return headers

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'fields': {name: field.to_dict() for name, field in self.fields.items()},
            'ensemble_keys': self.ensemble_keys,
            'computed_fields': [field.to_dict() for field in self.computed_fields]
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'HeaderMapping':
        """Create from dictionary."""
        mapping = cls()
        for name, field_data in data['fields'].items():
            mapping.add_field(HeaderField.from_dict(field_data))
        mapping.ensemble_keys = data.get('ensemble_keys', [])

        # Load computed fields
        for computed_data in data.get('computed_fields', []):
            mapping.add_computed_field(ComputedHeaderField.from_dict(computed_data))

        return mapping

    def save_to_file(self, filename: str):
        """Save mapping configuration to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filename: str) -> 'HeaderMapping':
        """Load mapping configuration from JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __repr__(self) -> str:
        return f"HeaderMapping(fields={len(self.fields)}, computed={len(self.computed_fields)}, ensemble_keys={self.ensemble_keys})"
