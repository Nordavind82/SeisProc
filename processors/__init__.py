"""
Processors package - seismic data processing operations.

Includes processor registry for multiprocess worker reconstruction.
"""
from typing import Dict, Type, Any

from .base_processor import BaseProcessor, ProgressCallback
from .bandpass_filter import BandpassFilter, ProcessingPipeline
from .tf_denoise import TFDenoise
from .stft_denoise import STFTDenoise
from .stockwell_denoise import StockwellDenoise
from .dwt_denoise import DWTDenoise, PYWT_AVAILABLE
from .gabor_denoise import GaborDenoise
from .emd_denoise import EMDDenoise, PYEMD_AVAILABLE
from .sst_denoise import SSTDenoise, SSQUEEZEPY_AVAILABLE
from .omp_denoise import OMPDenoise
from .denoise_3d import (
    Denoise3D,
    Denoise3DConfig,
    compute_3d_mad,
    get_available_headers,
    apply_denoise3d_design_mode,
    apply_denoise3d_application_mode,
)
from .agc import apply_agc_vectorized, remove_agc, apply_agc_to_gather
from .fk_filter import FKFilter
from .fkk_filter_gpu import FKKFilterGPU, FKKFilterCPU, get_fkk_filter
from .fkk_coordinate_filter import (
    FKKCoordinateFilter,
    FKKCoordinateConfig,
    apply_fkk_coordinate_filter,
    apply_fkk_design_mode,
    apply_fkk_application_mode,
)
from .gain_processor import GainProcessor
from .chunked_processor import ChunkedProcessor
from .spectral_analyzer import SpectralAnalyzer
from .mute_processor import MuteConfig, MuteProcessor
from .deconvolution import DeconvolutionProcessor, DeconConfig, apply_spiking_decon, apply_predictive_decon

# NMO and Stacking
from .nmo_processor import NMOProcessor, NMOConfig, apply_nmo_gather, compute_nmo_time
from .cdp_stacker import CDPStacker, StackConfig, StackResult, stack_cdp_gather, nmo_and_stack

# QC Engines
from .qc_stacking_engine import QCStackingEngine, QCStackingWorker, StackingProgress
from .qc_batch_engine import QCBatchEngine, QCBatchWorker, BatchProgress, BatchResult

# Migration processors
from .migration.kirchhoff_migrator import KirchhoffMigrator

# =============================================================================
# Processor Registry for Multiprocess Workers
# =============================================================================

# Registry mapping class names to processor classes
PROCESSOR_REGISTRY: Dict[str, Type[BaseProcessor]] = {
    'BandpassFilter': BandpassFilter,
    'TFDenoise': TFDenoise,
    'STFTDenoise': STFTDenoise,
    'StockwellDenoise': StockwellDenoise,
    'DWTDenoise': DWTDenoise,
    'GaborDenoise': GaborDenoise,
    'EMDDenoise': EMDDenoise,
    'SSTDenoise': SSTDenoise,
    'OMPDenoise': OMPDenoise,
    'Denoise3D': Denoise3D,
    'FKFilter': FKFilter,
    'GainProcessor': GainProcessor,
    'ChunkedProcessor': ChunkedProcessor,
    'SpectralAnalyzer': SpectralAnalyzer,
    'ProcessingPipeline': ProcessingPipeline,
    'KirchhoffMigrator': KirchhoffMigrator,
    'NMOProcessor': NMOProcessor,
    'CDPStacker': CDPStacker,
    'DeconvolutionProcessor': DeconvolutionProcessor,
}


def get_processor_class(name: str) -> Type[BaseProcessor]:
    """
    Get processor class by name from registry.

    Args:
        name: Processor class name (e.g., 'TFDenoise')

    Returns:
        Processor class

    Raises:
        KeyError: If processor name not found in registry
    """
    if name not in PROCESSOR_REGISTRY:
        raise KeyError(f"Unknown processor: {name}. Available: {list(PROCESSOR_REGISTRY.keys())}")
    return PROCESSOR_REGISTRY[name]


def create_processor(config: Dict[str, Any]) -> BaseProcessor:
    """
    Create processor instance from configuration dict.

    This is a convenience wrapper around BaseProcessor.from_dict()
    that can also use the registry for faster lookup.

    Args:
        config: Dictionary with 'class_name', 'module', and 'params' keys

    Returns:
        New processor instance

    Raises:
        ValueError: If config is invalid
    """
    return BaseProcessor.from_dict(config)


def register_processor(name: str, processor_class: Type[BaseProcessor]) -> None:
    """
    Register a custom processor class.

    Args:
        name: Name to register under
        processor_class: Processor class (must be BaseProcessor subclass)

    Raises:
        TypeError: If processor_class is not a BaseProcessor subclass
    """
    if not issubclass(processor_class, BaseProcessor):
        raise TypeError(f"{processor_class} is not a BaseProcessor subclass")
    PROCESSOR_REGISTRY[name] = processor_class


__all__ = [
    # Base classes
    'BaseProcessor',
    'ProgressCallback',
    # Processors
    'BandpassFilter',
    'ProcessingPipeline',
    'TFDenoise',
    'STFTDenoise',
    'StockwellDenoise',
    'DWTDenoise',
    'PYWT_AVAILABLE',
    'GaborDenoise',
    'EMDDenoise',
    'PYEMD_AVAILABLE',
    'SSTDenoise',
    'SSQUEEZEPY_AVAILABLE',
    'OMPDenoise',
    'Denoise3D',
    'Denoise3DConfig',
    'compute_3d_mad',
    'get_available_headers',
    'apply_denoise3d_design_mode',
    'apply_denoise3d_application_mode',
    'FKFilter',
    'FKKFilterGPU',
    'FKKFilterCPU',
    'get_fkk_filter',
    # FKK Coordinate Filter
    'FKKCoordinateFilter',
    'FKKCoordinateConfig',
    'apply_fkk_coordinate_filter',
    'apply_fkk_design_mode',
    'apply_fkk_application_mode',
    'GainProcessor',
    'ChunkedProcessor',
    'SpectralAnalyzer',
    # Mute
    'MuteConfig',
    'MuteProcessor',
    # Deconvolution
    'DeconvolutionProcessor',
    'DeconConfig',
    'apply_spiking_decon',
    'apply_predictive_decon',
    # NMO and Stacking
    'NMOProcessor',
    'NMOConfig',
    'apply_nmo_gather',
    'compute_nmo_time',
    'CDPStacker',
    'StackConfig',
    'StackResult',
    'stack_cdp_gather',
    'nmo_and_stack',
    # QC Engines
    'QCStackingEngine',
    'QCStackingWorker',
    'StackingProgress',
    'QCBatchEngine',
    'QCBatchWorker',
    'BatchProgress',
    'BatchResult',
    # Migration
    'KirchhoffMigrator',
    # AGC functions
    'apply_agc_vectorized',
    'remove_agc',
    'apply_agc_to_gather',
    # Registry functions
    'PROCESSOR_REGISTRY',
    'get_processor_class',
    'create_processor',
    'register_processor',
]
