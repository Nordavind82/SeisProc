"""
Header Mapping for PSTM

User-defined mapping between input file headers and application-required headers.
Supports validation, serialization, and auto-detection.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
import json
import logging

from models.header_schema import HeaderSchema, get_pstm_header_schema, HeaderRequirement

logger = logging.getLogger(__name__)


@dataclass
class HeaderMappingEntry:
    """
    Single header mapping entry.

    Attributes:
        schema_name: Internal schema header name (e.g., 'SOURCE_X')
        input_name: Header name in input file (e.g., 'SourceX')
        is_computed: Whether this header will be computed (not from file)
        transform: Optional transformation to apply (e.g., 'scale:0.1')
    """
    schema_name: str
    input_name: Optional[str] = None
    is_computed: bool = False
    transform: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'schema_name': self.schema_name,
            'input_name': self.input_name,
            'is_computed': self.is_computed,
            'transform': self.transform,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'HeaderMappingEntry':
        return cls(**d)


class HeaderMapping:
    """
    Complete header mapping configuration.

    Maps input file headers to schema-defined headers,
    handles computed headers, and validates completeness.

    Attributes:
        entries: Dictionary of mapping entries keyed by schema name
        coordinate_scalar: Global coordinate scalar (SEG-Y convention)
        schema: Header schema used for validation
    """

    def __init__(
        self,
        schema: Optional[HeaderSchema] = None,
        coordinate_scalar: float = 1.0,
    ):
        """
        Initialize header mapping.

        Args:
            schema: Header schema to use (defaults to PSTM schema)
            coordinate_scalar: Coordinate scalar multiplier
        """
        self.schema = schema or get_pstm_header_schema()
        self.coordinate_scalar = coordinate_scalar
        self.entries: Dict[str, HeaderMappingEntry] = {}
        self.metadata: Dict[str, Any] = {}

    def add_mapping(
        self,
        schema_name: str,
        input_name: str,
        transform: Optional[str] = None,
    ) -> None:
        """
        Add a direct header mapping.

        Args:
            schema_name: Schema header name
            input_name: Input file header name
            transform: Optional transformation string
        """
        if schema_name not in self.schema.headers:
            logger.warning(f"Header '{schema_name}' not in schema, adding anyway")

        self.entries[schema_name] = HeaderMappingEntry(
            schema_name=schema_name,
            input_name=input_name,
            is_computed=False,
            transform=transform,
        )
        logger.debug(f"Mapped '{input_name}' -> '{schema_name}'")

    def add_computed(self, schema_name: str) -> None:
        """
        Mark a header as computed (will be calculated from other headers).

        Args:
            schema_name: Schema header name to compute
        """
        header_def = self.schema.get_header(schema_name)
        if header_def and not header_def.can_compute:
            raise ValueError(f"Header '{schema_name}' cannot be computed")

        self.entries[schema_name] = HeaderMappingEntry(
            schema_name=schema_name,
            input_name=None,
            is_computed=True,
        )
        logger.debug(f"Marked '{schema_name}' as computed")

    def remove_mapping(self, schema_name: str) -> None:
        """Remove a header mapping."""
        if schema_name in self.entries:
            del self.entries[schema_name]

    def get_input_name(self, schema_name: str) -> Optional[str]:
        """Get input file header name for a schema header."""
        entry = self.entries.get(schema_name)
        if entry and not entry.is_computed:
            return entry.input_name
        return None

    def is_mapped(self, schema_name: str) -> bool:
        """Check if a schema header is mapped (directly or computed)."""
        return schema_name in self.entries

    def is_computed(self, schema_name: str) -> bool:
        """Check if a schema header is marked as computed."""
        entry = self.entries.get(schema_name)
        return entry is not None and entry.is_computed

    def get_mapped_input_headers(self) -> List[str]:
        """Get list of input headers that are mapped."""
        return [
            entry.input_name
            for entry in self.entries.values()
            if entry.input_name is not None
        ]

    def get_computed_headers(self) -> List[str]:
        """Get list of headers that will be computed."""
        return [
            name for name, entry in self.entries.items()
            if entry.is_computed
        ]

    def auto_detect(self, available_headers: List[str]) -> int:
        """
        Auto-detect header mappings from available headers.

        Args:
            available_headers: List of header names from input file

        Returns:
            Number of headers successfully mapped
        """
        auto_mapping = self.schema.auto_map_headers(available_headers)

        count = 0
        for schema_name, input_name in auto_mapping.items():
            if schema_name not in self.entries:
                self.add_mapping(schema_name, input_name)
                count += 1

        logger.info(f"Auto-detected {count} header mappings")
        return count

    def auto_add_computed(self) -> int:
        """
        Automatically add computed headers where possible.

        Returns:
            Number of computed headers added
        """
        count = 0

        for schema_name, header_def in self.schema.headers.items():
            if schema_name in self.entries:
                continue

            if not header_def.can_compute:
                continue

            # Check if dependencies are available
            deps_available = all(
                self.is_mapped(dep) for dep in header_def.compute_from
            )

            if deps_available:
                self.add_computed(schema_name)
                count += 1
                logger.debug(f"Auto-added computed header: {schema_name}")

        logger.info(f"Auto-added {count} computed headers")
        return count

    def validate(self) -> Dict[str, Any]:
        """
        Validate the mapping against schema requirements.

        Returns:
            Validation result dictionary
        """
        result = self.schema.validate_mapping(
            {name: entry.input_name or name for name, entry in self.entries.items()
             if not entry.is_computed},
            check_computable=True,
        )

        # Check computed header dependencies
        for schema_name, entry in self.entries.items():
            if entry.is_computed:
                header_def = self.schema.get_header(schema_name)
                if header_def:
                    for dep in header_def.compute_from:
                        if not self.is_mapped(dep):
                            result['valid'] = False
                            result['warnings'].append(
                                f"Computed header '{schema_name}' requires '{dep}' "
                                f"which is not mapped"
                            )

        return result

    def get_status_summary(self) -> Dict[str, Any]:
        """
        Get a summary of mapping status.

        Returns:
            Status dictionary with counts and lists
        """
        required = self.schema.get_required_headers()
        preferred = self.schema.get_preferred_headers()

        mapped_required = [h for h in required if self.is_mapped(h)]
        mapped_preferred = [h for h in preferred if self.is_mapped(h)]
        computed = self.get_computed_headers()

        return {
            'total_mapped': len(self.entries),
            'required_mapped': len(mapped_required),
            'required_total': len(required),
            'preferred_mapped': len(mapped_preferred),
            'preferred_total': len(preferred),
            'computed_count': len(computed),
            'missing_required': [h for h in required if not self.is_mapped(h)],
            'is_complete': len(mapped_required) == len(required),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize mapping to dictionary."""
        return {
            'entries': {
                name: entry.to_dict()
                for name, entry in self.entries.items()
            },
            'coordinate_scalar': self.coordinate_scalar,
            'metadata': self.metadata.copy(),
        }

    @classmethod
    def from_dict(
        cls,
        d: Dict[str, Any],
        schema: Optional[HeaderSchema] = None,
    ) -> 'HeaderMapping':
        """Deserialize mapping from dictionary."""
        mapping = cls(
            schema=schema,
            coordinate_scalar=d.get('coordinate_scalar', 1.0),
        )

        for name, entry_dict in d.get('entries', {}).items():
            mapping.entries[name] = HeaderMappingEntry.from_dict(entry_dict)

        mapping.metadata = d.get('metadata', {})
        return mapping

    def save(self, filepath: str) -> None:
        """Save mapping to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved header mapping to {filepath}")

    @classmethod
    def load(
        cls,
        filepath: str,
        schema: Optional[HeaderSchema] = None,
    ) -> 'HeaderMapping':
        """Load mapping from JSON file."""
        with open(filepath, 'r') as f:
            d = json.load(f)
        logger.info(f"Loaded header mapping from {filepath}")
        return cls.from_dict(d, schema)

    def __repr__(self) -> str:
        status = self.get_status_summary()
        return (
            f"HeaderMapping({status['total_mapped']} entries, "
            f"{status['required_mapped']}/{status['required_total']} required, "
            f"{status['computed_count']} computed)"
        )


# =============================================================================
# Factory Functions
# =============================================================================

def create_default_mapping(
    available_headers: Optional[List[str]] = None,
) -> HeaderMapping:
    """
    Create a default header mapping with auto-detection.

    Args:
        available_headers: Optional list of available input headers

    Returns:
        HeaderMapping with auto-detected mappings
    """
    mapping = HeaderMapping()

    if available_headers:
        mapping.auto_detect(available_headers)
        mapping.auto_add_computed()

    return mapping


def create_segy_mapping(
    use_standard_names: bool = True,
    coordinate_scalar: float = 1.0,
) -> HeaderMapping:
    """
    Create mapping for standard SEG-Y header names.

    Args:
        use_standard_names: Use segyio-style header names
        coordinate_scalar: Coordinate scalar

    Returns:
        HeaderMapping for SEG-Y files
    """
    mapping = HeaderMapping(coordinate_scalar=coordinate_scalar)

    if use_standard_names:
        # Standard segyio header names
        mapping.add_mapping('SOURCE_X', 'SourceX')
        mapping.add_mapping('SOURCE_Y', 'SourceY')
        mapping.add_mapping('RECEIVER_X', 'GroupX')
        mapping.add_mapping('RECEIVER_Y', 'GroupY')
        mapping.add_mapping('INLINE', 'INLINE_3D')
        mapping.add_mapping('CROSSLINE', 'CROSSLINE_3D')
        mapping.add_mapping('SHOT_NUMBER', 'FieldRecord')
        mapping.add_mapping('CHANNEL', 'TraceNumber')
        mapping.add_mapping('CDP', 'CDP')

    # Add computed headers
    mapping.add_computed('OFFSET')
    mapping.add_computed('AZIMUTH')
    mapping.add_computed('CDP_X')
    mapping.add_computed('CDP_Y')

    return mapping


def get_common_header_names() -> Dict[str, List[str]]:
    """
    Get dictionary of common header name variations.

    Returns:
        Dictionary mapping schema names to lists of common variations
    """
    return {
        'SOURCE_X': ['SourceX', 'SX', 'sx', 'SRCX', 'source_x', 'ShotX', 'Source X'],
        'SOURCE_Y': ['SourceY', 'SY', 'sy', 'SRCY', 'source_y', 'ShotY', 'Source Y'],
        'RECEIVER_X': ['GroupX', 'GX', 'gx', 'GRPX', 'receiver_x', 'RecX', 'ReceiverX', 'Group X'],
        'RECEIVER_Y': ['GroupY', 'GY', 'gy', 'GRPY', 'receiver_y', 'RecY', 'ReceiverY', 'Group Y'],
        'OFFSET': ['offset', 'Offset', 'OFFSET', 'SrcRecDist', 'Distance'],
        'AZIMUTH': ['azimuth', 'Azimuth', 'AZIMUTH', 'AZ', 'SrcRecAzim'],
        'INLINE': ['INLINE_3D', 'Inline', 'IL', 'inline', 'InlineNumber', 'iline'],
        'CROSSLINE': ['CROSSLINE_3D', 'Crossline', 'XL', 'crossline', 'XLINE', 'xline'],
        'CDP': ['CDP', 'cdp', 'EnsembleNumber', 'CMP', 'cmp'],
        'SHOT_NUMBER': ['FieldRecord', 'FFID', 'ShotNumber', 'SP', 'shot_number'],
    }
