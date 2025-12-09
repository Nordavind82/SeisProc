"""
Header Schema for PSTM

Defines required and optional headers for pre-stack time migration,
including validation and computed header definitions.
"""

from typing import Dict, Any, List, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)


class HeaderRequirement(Enum):
    """Header requirement level."""
    REQUIRED = "required"        # Must be present or computable
    PREFERRED = "preferred"      # Should be present, can compute if missing
    OPTIONAL = "optional"        # Nice to have, not critical


class HeaderDataType(Enum):
    """Expected data type for header values."""
    INT16 = "int16"
    INT32 = "int32"
    FLOAT32 = "float32"
    FLOAT64 = "float64"


@dataclass
class HeaderDefinition:
    """
    Definition of a single header field.

    Attributes:
        name: Internal standardized name (e.g., 'SOURCE_X')
        description: Human-readable description
        requirement: Required, preferred, or optional
        data_type: Expected data type
        unit: Physical unit (e.g., 'meters', 'degrees')
        standard_names: List of common names in various SEG-Y implementations
        can_compute: Whether this header can be computed from other headers
        compute_from: List of headers needed to compute this one
        default_value: Default value if not available
    """
    name: str
    description: str
    requirement: HeaderRequirement = HeaderRequirement.OPTIONAL
    data_type: HeaderDataType = HeaderDataType.FLOAT32
    unit: str = ""
    standard_names: List[str] = field(default_factory=list)
    can_compute: bool = False
    compute_from: List[str] = field(default_factory=list)
    default_value: Optional[float] = None

    def matches_name(self, header_name: str) -> bool:
        """Check if a header name matches this definition."""
        name_lower = header_name.lower().replace('_', '').replace(' ', '')

        # Check exact match
        if header_name == self.name:
            return True

        # Check standard names
        for std_name in self.standard_names:
            std_lower = std_name.lower().replace('_', '').replace(' ', '')
            if name_lower == std_lower:
                return True

        return False


class HeaderSchema:
    """
    Schema defining all headers required/used by PSTM.

    Provides:
    - Standard header definitions
    - Validation of available headers
    - Computation rules for derived headers
    """

    def __init__(self):
        """Initialize header schema with standard PSTM headers."""
        self.headers: Dict[str, HeaderDefinition] = {}
        self._build_standard_schema()

    def _build_standard_schema(self):
        """Build the standard PSTM header schema."""

        # Source coordinates
        self.add_header(HeaderDefinition(
            name='SOURCE_X',
            description='Source X coordinate',
            requirement=HeaderRequirement.REQUIRED,
            data_type=HeaderDataType.FLOAT64,
            unit='meters',
            standard_names=['SourceX', 'SX', 'sx', 'SRCX', 'source_x', 'ShotX'],
        ))

        self.add_header(HeaderDefinition(
            name='SOURCE_Y',
            description='Source Y coordinate',
            requirement=HeaderRequirement.REQUIRED,
            data_type=HeaderDataType.FLOAT64,
            unit='meters',
            standard_names=['SourceY', 'SY', 'sy', 'SRCY', 'source_y', 'ShotY'],
        ))

        # Receiver (Group) coordinates
        self.add_header(HeaderDefinition(
            name='RECEIVER_X',
            description='Receiver X coordinate',
            requirement=HeaderRequirement.REQUIRED,
            data_type=HeaderDataType.FLOAT64,
            unit='meters',
            standard_names=['GroupX', 'GX', 'gx', 'GRPX', 'receiver_x', 'RecX', 'ReceiverX'],
        ))

        self.add_header(HeaderDefinition(
            name='RECEIVER_Y',
            description='Receiver Y coordinate',
            requirement=HeaderRequirement.REQUIRED,
            data_type=HeaderDataType.FLOAT64,
            unit='meters',
            standard_names=['GroupY', 'GY', 'gy', 'GRPY', 'receiver_y', 'RecY', 'ReceiverY'],
        ))

        # Derived geometry headers
        self.add_header(HeaderDefinition(
            name='OFFSET',
            description='Source-receiver offset',
            requirement=HeaderRequirement.PREFERRED,
            data_type=HeaderDataType.FLOAT32,
            unit='meters',
            standard_names=['offset', 'Offset', 'OFFSET', 'SrcRecDist'],
            can_compute=True,
            compute_from=['SOURCE_X', 'SOURCE_Y', 'RECEIVER_X', 'RECEIVER_Y'],
        ))

        self.add_header(HeaderDefinition(
            name='AZIMUTH',
            description='Source-to-receiver azimuth',
            requirement=HeaderRequirement.PREFERRED,
            data_type=HeaderDataType.FLOAT32,
            unit='degrees',
            standard_names=['azimuth', 'Azimuth', 'AZIMUTH', 'AZ', 'SrcRecAzim'],
            can_compute=True,
            compute_from=['SOURCE_X', 'SOURCE_Y', 'RECEIVER_X', 'RECEIVER_Y'],
        ))

        self.add_header(HeaderDefinition(
            name='CDP_X',
            description='CDP X coordinate',
            requirement=HeaderRequirement.OPTIONAL,
            data_type=HeaderDataType.FLOAT64,
            unit='meters',
            standard_names=['CDP_X', 'CDPX', 'cdp_x', 'MidpointX', 'CMP_X'],
            can_compute=True,
            compute_from=['SOURCE_X', 'RECEIVER_X'],
        ))

        self.add_header(HeaderDefinition(
            name='CDP_Y',
            description='CDP Y coordinate',
            requirement=HeaderRequirement.OPTIONAL,
            data_type=HeaderDataType.FLOAT64,
            unit='meters',
            standard_names=['CDP_Y', 'CDPY', 'cdp_y', 'MidpointY', 'CMP_Y'],
            can_compute=True,
            compute_from=['SOURCE_Y', 'RECEIVER_Y'],
        ))

        # Survey indexing headers
        self.add_header(HeaderDefinition(
            name='INLINE',
            description='Inline number',
            requirement=HeaderRequirement.OPTIONAL,
            data_type=HeaderDataType.INT32,
            unit='',
            standard_names=['INLINE', 'Inline', 'IL', 'inline', 'InlineNumber',
                          'INLINE_3D', 'Inline3D', 'iline'],
        ))

        self.add_header(HeaderDefinition(
            name='CROSSLINE',
            description='Crossline number',
            requirement=HeaderRequirement.OPTIONAL,
            data_type=HeaderDataType.INT32,
            unit='',
            standard_names=['CROSSLINE', 'Crossline', 'XL', 'crossline', 'XLINE',
                          'CrosslineNumber', 'XLINE_3D', 'Crossline3D', 'xline'],
        ))

        # Shot/receiver indexing
        self.add_header(HeaderDefinition(
            name='SHOT_NUMBER',
            description='Shot point number',
            requirement=HeaderRequirement.OPTIONAL,
            data_type=HeaderDataType.INT32,
            unit='',
            standard_names=['FieldRecord', 'FFID', 'ShotNumber', 'SP', 'shot_number',
                          'SourcePoint', 'SRCID'],
        ))

        self.add_header(HeaderDefinition(
            name='CHANNEL',
            description='Channel/trace number within shot',
            requirement=HeaderRequirement.OPTIONAL,
            data_type=HeaderDataType.INT32,
            unit='',
            standard_names=['TraceNumber', 'CHAN', 'Channel', 'channel',
                          'TraceInFile', 'TraceId'],
        ))

        # Coordinate scalar
        self.add_header(HeaderDefinition(
            name='COORD_SCALAR',
            description='Coordinate scalar (SEG-Y convention)',
            requirement=HeaderRequirement.OPTIONAL,
            data_type=HeaderDataType.INT16,
            unit='',
            standard_names=['SourceGroupScalar', 'CoordScalar', 'scalar_co',
                          'ScalarCoordinates', 'CoordinateScalar'],
            default_value=1.0,
        ))

        # Elevation headers
        self.add_header(HeaderDefinition(
            name='SOURCE_ELEV',
            description='Source elevation/depth',
            requirement=HeaderRequirement.OPTIONAL,
            data_type=HeaderDataType.FLOAT32,
            unit='meters',
            standard_names=['SourceSurfaceElevation', 'SrcElev', 'SourceElev',
                          'source_elev', 'ShotElev'],
        ))

        self.add_header(HeaderDefinition(
            name='RECEIVER_ELEV',
            description='Receiver elevation/depth',
            requirement=HeaderRequirement.OPTIONAL,
            data_type=HeaderDataType.FLOAT32,
            unit='meters',
            standard_names=['ReceiverGroupElevation', 'RecElev', 'GroupElev',
                          'receiver_elev', 'GrpElev'],
        ))

        # Trace sorting headers
        self.add_header(HeaderDefinition(
            name='CDP',
            description='CDP number (ensemble key)',
            requirement=HeaderRequirement.OPTIONAL,
            data_type=HeaderDataType.INT32,
            unit='',
            standard_names=['CDP', 'cdp', 'EnsembleNumber', 'CMP', 'cmp'],
        ))

        self.add_header(HeaderDefinition(
            name='TRACE_IN_ENSEMBLE',
            description='Trace number within CDP/ensemble',
            requirement=HeaderRequirement.OPTIONAL,
            data_type=HeaderDataType.INT32,
            unit='',
            standard_names=['TraceInEnsemble', 'TRACF', 'TraceNumWithinEnsemble',
                          'CDPTrace'],
        ))

    def add_header(self, header_def: HeaderDefinition):
        """Add a header definition to the schema."""
        self.headers[header_def.name] = header_def

    def get_header(self, name: str) -> Optional[HeaderDefinition]:
        """Get header definition by name."""
        return self.headers.get(name)

    def get_required_headers(self) -> List[str]:
        """Get list of required header names."""
        return [
            name for name, hdr in self.headers.items()
            if hdr.requirement == HeaderRequirement.REQUIRED
        ]

    def get_preferred_headers(self) -> List[str]:
        """Get list of preferred header names."""
        return [
            name for name, hdr in self.headers.items()
            if hdr.requirement == HeaderRequirement.PREFERRED
        ]

    def get_computable_headers(self) -> List[str]:
        """Get list of headers that can be computed."""
        return [
            name for name, hdr in self.headers.items()
            if hdr.can_compute
        ]

    def find_matching_header(self, input_name: str) -> Optional[str]:
        """
        Find schema header that matches an input header name.

        Args:
            input_name: Header name from input file

        Returns:
            Schema header name if found, None otherwise
        """
        for name, header_def in self.headers.items():
            if header_def.matches_name(input_name):
                return name
        return None

    def auto_map_headers(
        self,
        available_headers: List[str]
    ) -> Dict[str, str]:
        """
        Automatically map available headers to schema headers.

        Args:
            available_headers: List of header names from input file

        Returns:
            Dictionary mapping schema names to input names
        """
        mapping = {}

        for input_name in available_headers:
            schema_name = self.find_matching_header(input_name)
            if schema_name and schema_name not in mapping:
                mapping[schema_name] = input_name
                logger.debug(f"Auto-mapped '{input_name}' -> '{schema_name}'")

        return mapping

    def validate_mapping(
        self,
        mapping: Dict[str, str],
        check_computable: bool = True,
    ) -> Dict[str, Any]:
        """
        Validate a header mapping against schema requirements.

        Args:
            mapping: Dictionary mapping schema names to input names
            check_computable: If True, consider computable headers as satisfying requirements

        Returns:
            Validation result dictionary
        """
        result = {
            'valid': True,
            'missing_required': [],
            'missing_preferred': [],
            'will_compute': [],
            'warnings': [],
        }

        mapped_headers = set(mapping.keys())

        for name, header_def in self.headers.items():
            if name in mapped_headers:
                continue

            if header_def.can_compute and check_computable:
                # Check if we can compute this header
                can_compute = all(
                    dep in mapped_headers or dep in result['will_compute']
                    for dep in header_def.compute_from
                )
                if can_compute:
                    result['will_compute'].append(name)
                    continue

            if header_def.requirement == HeaderRequirement.REQUIRED:
                result['missing_required'].append(name)
                result['valid'] = False
            elif header_def.requirement == HeaderRequirement.PREFERRED:
                result['missing_preferred'].append(name)
                result['warnings'].append(
                    f"Preferred header '{name}' not mapped and cannot be computed"
                )

        return result

    def to_dict(self) -> Dict[str, Any]:
        """Serialize schema to dictionary."""
        return {
            name: {
                'description': hdr.description,
                'requirement': hdr.requirement.value,
                'data_type': hdr.data_type.value,
                'unit': hdr.unit,
                'standard_names': hdr.standard_names,
                'can_compute': hdr.can_compute,
                'compute_from': hdr.compute_from,
            }
            for name, hdr in self.headers.items()
        }


# Global schema instance
_pstm_schema: Optional[HeaderSchema] = None


def get_pstm_header_schema() -> HeaderSchema:
    """Get the global PSTM header schema instance."""
    global _pstm_schema
    if _pstm_schema is None:
        _pstm_schema = HeaderSchema()
    return _pstm_schema
