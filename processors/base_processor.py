"""
Base processor class - abstract interface for all seismic processing operations.
This ensures all processors follow the same contract.

Supports serialization for multiprocess parallel processing.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, Type
import sys
import importlib
from models.seismic_data import SeismicData

# Type alias for progress callbacks: (current, total, message) -> None
ProgressCallback = Callable[[int, int, str], None]


class BaseProcessor(ABC):
    """
    Abstract base class for all seismic processors.

    All processors must:
    1. Be immutable (don't modify input data)
    2. Return new SeismicData object
    3. Validate parameters in __init__
    4. Provide clear parameter description
    """

    def __init__(self, **params):
        """
        Initialize processor with parameters.

        Args:
            **params: Processor-specific parameters
        """
        self.params = params
        self._progress_callback: Optional[ProgressCallback] = None
        self._validate_params()

    def set_progress_callback(self, callback: Optional[ProgressCallback]) -> 'BaseProcessor':
        """
        Set progress callback for processing status updates.

        Args:
            callback: Function(current, total, message) to receive progress updates.
                     Set to None to disable progress reporting.

        Returns:
            self for method chaining
        """
        self._progress_callback = callback
        return self

    def _report_progress(self, current: int, total: int, message: str = ""):
        """
        Report progress to callback if set.

        Args:
            current: Current progress value (e.g., traces processed)
            total: Total items to process
            message: Optional status message
        """
        if self._progress_callback is not None:
            self._progress_callback(current, total, message)

    @abstractmethod
    def _validate_params(self):
        """Validate processor parameters. Raise ValueError if invalid."""
        pass

    @abstractmethod
    def process(self, data: SeismicData) -> SeismicData:
        """
        Process seismic data.

        Args:
            data: Input seismic data

        Returns:
            Processed seismic data (new object, input unchanged)
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Get human-readable description of this processor and its parameters."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.params})"

    # =========================================================================
    # Serialization for Multiprocess Workers
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize processor configuration for multiprocess transfer.

        Returns:
            Dictionary with class info and parameters that can be pickled
            and used to reconstruct the processor in a worker process.
        """
        return {
            'class_name': self.__class__.__name__,
            'module': self.__class__.__module__,
            'params': self.params.copy()
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'BaseProcessor':
        """
        Reconstruct processor from serialized configuration.

        Args:
            config: Dictionary from to_dict() containing class info and params

        Returns:
            New processor instance with same configuration

        Raises:
            ValueError: If config is invalid or class not found
        """
        try:
            module_name = config['module']
            class_name = config['class_name']
            params = config['params']

            # Import the module containing the processor class
            module = importlib.import_module(module_name)

            # Get the processor class
            processor_class = getattr(module, class_name)

            # Verify it's a BaseProcessor subclass
            if not issubclass(processor_class, BaseProcessor):
                raise ValueError(f"{class_name} is not a BaseProcessor subclass")

            # Create new instance with saved parameters
            return processor_class(**params)

        except KeyError as e:
            raise ValueError(f"Invalid processor config - missing key: {e}")
        except ImportError as e:
            raise ValueError(f"Cannot import processor module '{config.get('module')}': {e}")
        except AttributeError as e:
            raise ValueError(f"Processor class '{config.get('class_name')}' not found: {e}")
