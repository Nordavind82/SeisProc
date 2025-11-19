"""
Computed header functionality for SEGY import.
Allows creation of new headers from existing ones using math equations.
"""

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Set, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class ComputedHeaderField:
    """
    Represents a computed header field created from existing headers using a math equation.

    Attributes:
        name: Name of the computed header
        expression: Mathematical expression (e.g., "round(receiver_station / 1000)")
        description: Human-readable description
        format: Output format ('i'=int32, 'f'=float32)
    """
    name: str
    expression: str
    description: str = ""
    format: str = 'f'  # Default to float

    def __post_init__(self):
        """Validate the field configuration."""
        if not self.name:
            raise ValueError("Computed header name cannot be empty")
        if not self.expression:
            raise ValueError(f"Expression for computed header '{self.name}' cannot be empty")
        if self.format not in ('i', 'f', 'h'):
            raise ValueError(f"Invalid format '{self.format}' for computed header '{self.name}'")

    def get_dependencies(self) -> Set[str]:
        """
        Extract header names referenced in the expression.

        Returns:
            Set of header names used in the expression
        """
        # Match valid Python identifiers (header names)
        # Exclude math function names and keywords
        excluded = {
            'abs', 'min', 'max', 'round', 'floor', 'ceil', 'sqrt',
            'sin', 'cos', 'tan', 'atan2', 'pi', 'e'
        }

        # Find all identifiers in the expression
        identifiers = set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', self.expression))

        # Remove excluded math functions
        dependencies = identifiers - excluded

        return dependencies

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'expression': self.expression,
            'description': self.description,
            'format': self.format
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComputedHeaderField':
        """Create from dictionary."""
        return cls(
            name=data['name'],
            expression=data['expression'],
            description=data.get('description', ''),
            format=data.get('format', 'f')
        )


class ComputedHeaderEvaluator:
    """
    Safely evaluates mathematical expressions for computed headers.
    Supports basic math operations, rounding, and common math functions.
    """

    # Allowed functions for safe evaluation
    SAFE_FUNCTIONS = {
        'abs': abs,
        'min': min,
        'max': max,
        'round': round,
        'floor': math.floor,
        'ceil': math.ceil,
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'atan2': math.atan2,
        'pi': math.pi,
        'e': math.e,
    }

    def __init__(self):
        self.error_stats = {
            'total_errors': 0,
            'errors_by_field': {},
            'error_traces': []  # List of (trace_idx, field_name, error_msg, headers)
        }

    def evaluate(self, expression: str, headers: Dict[str, Any],
                 field_name: str = "", trace_idx: int = -1) -> Optional[float]:
        """
        Safely evaluate a mathematical expression with header values.

        Args:
            expression: Mathematical expression to evaluate
            headers: Dictionary of header names to values
            field_name: Name of the computed field (for error reporting)
            trace_idx: Trace index (for error reporting)

        Returns:
            Computed value or 0.0 if evaluation fails
        """
        try:
            # Create a safe namespace with allowed functions and header values
            namespace = {**self.SAFE_FUNCTIONS, **headers}

            # Evaluate the expression
            result = eval(expression, {"__builtins__": {}}, namespace)

            # Handle None or invalid results
            if result is None or (isinstance(result, float) and (math.isnan(result) or math.isinf(result))):
                self._record_error(field_name, trace_idx, "Result is None/NaN/Inf", headers)
                return 0.0

            return float(result)

        except ZeroDivisionError as e:
            self._record_error(field_name, trace_idx, f"Division by zero: {e}", headers)
            return 0.0
        except KeyError as e:
            self._record_error(field_name, trace_idx, f"Missing header: {e}", headers)
            return 0.0
        except Exception as e:
            self._record_error(field_name, trace_idx, f"Evaluation error: {e}", headers)
            return 0.0

    def _record_error(self, field_name: str, trace_idx: int, error_msg: str,
                      headers: Dict[str, Any]):
        """Record error for statistics and reporting."""
        self.error_stats['total_errors'] += 1

        if field_name not in self.error_stats['errors_by_field']:
            self.error_stats['errors_by_field'][field_name] = 0
        self.error_stats['errors_by_field'][field_name] += 1

        # Store first 100 error traces for detailed reporting
        if len(self.error_stats['error_traces']) < 100:
            self.error_stats['error_traces'].append({
                'trace_idx': trace_idx,
                'field_name': field_name,
                'error': error_msg,
                'headers': dict(headers)  # Copy for safety
            })

    def format_value(self, value: float, format_type: str) -> Any:
        """
        Format the computed value according to the specified format.

        Args:
            value: Computed value
            format_type: Format type ('i'=int32, 'h'=int16, 'f'=float32)

        Returns:
            Formatted value
        """
        if format_type == 'i':
            # 32-bit integer
            return int(round(value))
        elif format_type == 'h':
            # 16-bit integer
            return int(round(value))
        else:  # 'f'
            # 32-bit float
            return float(value)

    def get_error_summary(self) -> str:
        """
        Generate a summary of evaluation errors.

        Returns:
            Formatted error summary string
        """
        if self.error_stats['total_errors'] == 0:
            return "No computation errors"

        lines = [
            f"\n{'='*60}",
            f"Computed Header Evaluation Errors: {self.error_stats['total_errors']} total",
            f"{'='*60}"
        ]

        # Errors by field
        if self.error_stats['errors_by_field']:
            lines.append("\nErrors by field:")
            for field, count in sorted(self.error_stats['errors_by_field'].items()):
                lines.append(f"  {field}: {count} errors")

        # Detailed error traces (first 10)
        if self.error_stats['error_traces']:
            lines.append(f"\nFirst {min(10, len(self.error_stats['error_traces']))} error traces:")
            for i, error_info in enumerate(self.error_stats['error_traces'][:10]):
                trace_idx = error_info['trace_idx']
                field_name = error_info['field_name']
                error = error_info['error']
                headers = error_info['headers']

                lines.append(f"\n  [{i+1}] Trace {trace_idx}, Field '{field_name}':")
                lines.append(f"      Error: {error}")

                # Show relevant headers for context
                if headers:
                    header_str = ", ".join([f"{k}={v}" for k, v in list(headers.items())[:5]])
                    lines.append(f"      Headers: {header_str}")

        lines.append(f"{'='*60}\n")
        return "\n".join(lines)

    def reset_stats(self):
        """Reset error statistics."""
        self.error_stats = {
            'total_errors': 0,
            'errors_by_field': {},
            'error_traces': []
        }


class ComputedHeaderProcessor:
    """
    Processes computed headers with dependency resolution and chaining support.
    """

    def __init__(self, computed_fields: List[ComputedHeaderField]):
        """
        Initialize processor with computed fields.

        Args:
            computed_fields: List of computed header field definitions
        """
        self.computed_fields = computed_fields
        self.evaluator = ComputedHeaderEvaluator()
        self.execution_order = []

        # Build execution order based on dependencies
        self._resolve_dependencies()

    def _resolve_dependencies(self):
        """
        Resolve dependencies and determine execution order for computed headers.
        Uses topological sort to handle chained computations.
        """
        # Build dependency graph
        dependency_graph = {}
        for field in self.computed_fields:
            dependency_graph[field.name] = field.get_dependencies()

        # Topological sort
        visited = set()
        temp_visited = set()
        self.execution_order = []

        def visit(field_name: str, field: ComputedHeaderField):
            if field_name in temp_visited:
                raise ValueError(f"Circular dependency detected involving '{field_name}'")
            if field_name in visited:
                return

            temp_visited.add(field_name)

            # Visit dependencies first
            deps = dependency_graph.get(field_name, set())
            for dep in deps:
                # Find the field for this dependency
                dep_field = next((f for f in self.computed_fields if f.name == dep), None)
                if dep_field:
                    visit(dep, dep_field)

            temp_visited.remove(field_name)
            visited.add(field_name)
            self.execution_order.append(field)

        # Visit all fields
        for field in self.computed_fields:
            if field.name not in visited:
                visit(field.name, field)

        logger.info(f"Computed header execution order: {[f.name for f in self.execution_order]}")

    def compute_headers(self, raw_headers: Dict[str, Any],
                       trace_idx: int = -1) -> Dict[str, Any]:
        """
        Compute all computed headers for a single trace.

        Args:
            raw_headers: Dictionary of raw header values from SEGY
            trace_idx: Trace index (for error reporting)

        Returns:
            Dictionary of computed header values
        """
        # Start with raw headers as the base
        all_headers = dict(raw_headers)
        computed = {}

        # Execute in dependency order
        for field in self.execution_order:
            value = self.evaluator.evaluate(
                field.expression,
                all_headers,
                field.name,
                trace_idx
            )

            # Format the value
            formatted_value = self.evaluator.format_value(value, field.format)

            # Store computed value
            computed[field.name] = formatted_value

            # Make available for subsequent computations (chaining)
            all_headers[field.name] = formatted_value

        return computed

    def compute_headers_batch(self, raw_headers_list: List[Dict[str, Any]],
                             start_trace_idx: int = 0) -> List[Dict[str, Any]]:
        """
        Compute headers for a batch of traces.

        Args:
            raw_headers_list: List of raw header dictionaries
            start_trace_idx: Starting trace index for error reporting

        Returns:
            List of dictionaries containing both raw and computed headers
        """
        result = []

        for i, raw_headers in enumerate(raw_headers_list):
            trace_idx = start_trace_idx + i

            # Compute headers for this trace
            computed = self.compute_headers(raw_headers, trace_idx)

            # Merge raw and computed headers
            combined = {**raw_headers, **computed}
            result.append(combined)

        return result

    def get_error_summary(self) -> str:
        """Get error summary from evaluator."""
        return self.evaluator.get_error_summary()

    def reset_stats(self):
        """Reset error statistics."""
        self.evaluator.reset_stats()

    def get_computed_field_names(self) -> List[str]:
        """Get list of computed field names."""
        return [field.name for field in self.computed_fields]
