"""
Base processor class - abstract interface for all seismic processing operations.
This ensures all processors follow the same contract.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
import sys
from models.seismic_data import SeismicData


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
        self._validate_params()

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
