"""
CDP Stacker - Stack NMO-corrected CDP gathers

Provides several stacking methods:
- Mean stack (standard)
- Median stack (robust to outliers)
- Weighted stack (by fold or SNR)

Features:
- Integrates with NMOProcessor for combined NMO+stack
- Computes fold and output headers
- Supports minimum fold cutoff
- Vectorized implementation

Usage:
    from processors.cdp_stacker import CDPStacker, StackConfig
    from processors.nmo_processor import NMOProcessor, NMOConfig

    stack_config = StackConfig(method='mean', min_fold=3)
    stacker = CDPStacker(stack_config)

    # Stack a single gather
    stacked_trace, fold = stacker.stack_gather(nmo_corrected_traces)

    # Or with NMO in one step
    stacker_with_nmo = CDPStacker(stack_config, nmo_processor=nmo_proc)
    stacked, fold = stacker_with_nmo.stack_gather_with_nmo(traces, offsets, dt_ms)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List, Union
import logging

from models.seismic_data import SeismicData
from models.velocity_model import VelocityModel
from processors.base_processor import BaseProcessor

logger = logging.getLogger(__name__)


@dataclass
class StackConfig:
    """
    Configuration for CDP stacking.

    Attributes:
        method: Stacking method ('mean', 'median', 'weighted')
        min_fold: Minimum fold required (samples with lower fold are zeroed)
        normalize: Whether to normalize output by fold
        weight_by_offset: For weighted stack, weight by inverse offset
        preserve_amplitude: Scale output to match input amplitude level
    """
    method: str = 'mean'
    min_fold: int = 1
    normalize: bool = True
    weight_by_offset: bool = False
    preserve_amplitude: bool = False

    def __post_init__(self):
        valid_methods = ['mean', 'median', 'weighted']
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got {self.method}")
        if self.min_fold < 1:
            raise ValueError(f"min_fold must be >= 1, got {self.min_fold}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'method': self.method,
            'min_fold': self.min_fold,
            'normalize': self.normalize,
            'weight_by_offset': self.weight_by_offset,
            'preserve_amplitude': self.preserve_amplitude,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'StackConfig':
        """Deserialize from dictionary."""
        return cls(
            method=d.get('method', 'mean'),
            min_fold=d.get('min_fold', 1),
            normalize=d.get('normalize', True),
            weight_by_offset=d.get('weight_by_offset', False),
            preserve_amplitude=d.get('preserve_amplitude', False),
        )


@dataclass
class StackResult:
    """
    Result of stacking operation.

    Attributes:
        trace: Stacked trace (n_samples,)
        fold: Fold at each sample (n_samples,)
        cdp: CDP number
        mean_offset: Mean absolute offset of contributing traces
        n_traces: Number of input traces
        rms_amplitude: RMS amplitude of stacked trace
    """
    trace: np.ndarray
    fold: np.ndarray
    cdp: Optional[int] = None
    mean_offset: float = 0.0
    n_traces: int = 0
    rms_amplitude: float = 0.0


class CDPStacker(BaseProcessor):
    """
    CDP stacker for seismic gathers.

    Stacks multiple traces into a single output trace using various methods.
    Optionally integrates NMO correction before stacking.

    Args:
        config: StackConfig with stacking parameters
        nmo_processor: Optional NMOProcessor for integrated NMO+stack

    Example:
        >>> config = StackConfig(method='mean', min_fold=3)
        >>> stacker = CDPStacker(config)
        >>> result = stacker.stack_gather(nmo_traces)
        >>> stacked_trace = result.trace
    """

    def __init__(
        self,
        config: Optional[StackConfig] = None,
        nmo_processor: Optional['NMOProcessor'] = None,
        **params
    ):
        # Handle config from params
        if config is None and 'config' in params:
            config = params.pop('config')

        self.config = config or StackConfig()
        self.nmo_processor = nmo_processor

        params['config'] = self.config.to_dict()
        params['has_nmo'] = nmo_processor is not None

        super().__init__(**params)

    def _validate_params(self):
        """Validate parameters."""
        pass

    def get_description(self) -> str:
        """Get human-readable description."""
        desc = f"CDP Stacker ({self.config.method}, min_fold={self.config.min_fold})"
        if self.nmo_processor:
            desc += "\n  with NMO correction"
        return desc

    def process(self, data: SeismicData) -> SeismicData:
        """
        Process SeismicData by stacking traces.

        Note: This stacks ALL traces in the data into a single trace.
        For gather-by-gather stacking, use stack_gather() directly.

        Args:
            data: Input SeismicData

        Returns:
            SeismicData with single stacked trace
        """
        result = self.stack_gather(data.traces)

        return SeismicData(
            traces=result.trace.reshape(-1, 1),
            sample_interval=data.sample_interval,
            start_time=data.start_time,
        )

    def stack_gather(
        self,
        traces: np.ndarray,
        offsets: Optional[np.ndarray] = None,
        mute_mask: Optional[np.ndarray] = None,
    ) -> StackResult:
        """
        Stack a single CDP gather.

        Args:
            traces: 2D array (n_samples, n_traces)
            offsets: Optional 1D array of offsets for weighted stacking
            mute_mask: Optional boolean mask (n_samples, n_traces) where True = dead

        Returns:
            StackResult with stacked trace and metadata
        """
        n_samples, n_traces = traces.shape

        # Create live mask (inverse of mute)
        if mute_mask is not None:
            live_mask = ~mute_mask
        else:
            # Consider zero traces as dead
            live_mask = np.abs(traces) > 1e-30

        # Compute fold (number of live traces at each sample)
        fold = np.sum(live_mask, axis=1).astype(np.int32)

        # Stack based on method
        if self.config.method == 'mean':
            stacked = self._mean_stack(traces, live_mask, fold)
        elif self.config.method == 'median':
            stacked = self._median_stack(traces, live_mask)
        elif self.config.method == 'weighted':
            weights = self._compute_weights(traces, offsets, live_mask)
            stacked = self._weighted_stack(traces, weights, live_mask)
        else:
            raise ValueError(f"Unknown stack method: {self.config.method}")

        # Apply minimum fold cutoff
        if self.config.min_fold > 1:
            stacked[fold < self.config.min_fold] = 0.0

        # Compute statistics
        mean_offset = float(np.mean(np.abs(offsets))) if offsets is not None else 0.0
        rms_amplitude = float(np.sqrt(np.mean(stacked**2)))

        return StackResult(
            trace=stacked.astype(np.float32),
            fold=fold,
            n_traces=n_traces,
            mean_offset=mean_offset,
            rms_amplitude=rms_amplitude,
        )

    def stack_gather_with_nmo(
        self,
        traces: np.ndarray,
        offsets: np.ndarray,
        sample_interval_ms: float,
        cdp: Optional[int] = None,
    ) -> StackResult:
        """
        Apply NMO correction and stack in one operation.

        Requires nmo_processor to be set.

        Args:
            traces: 2D array (n_samples, n_traces)
            offsets: 1D array of offsets in meters
            sample_interval_ms: Sample interval in milliseconds
            cdp: Optional CDP number for 2D velocity models

        Returns:
            StackResult with stacked trace
        """
        if self.nmo_processor is None:
            raise ValueError("NMO processor not set")

        # Apply NMO
        nmo_traces = self.nmo_processor.apply_nmo(
            traces, offsets, sample_interval_ms, cdp
        )

        # Get stretch mute mask
        mute_mask = self.nmo_processor.compute_stretch_mute_mask(
            offsets, sample_interval_ms, traces.shape[0], cdp
        )

        # Stack
        result = self.stack_gather(nmo_traces, offsets, mute_mask)
        result.cdp = cdp

        return result

    def _mean_stack(
        self,
        traces: np.ndarray,
        live_mask: np.ndarray,
        fold: np.ndarray,
    ) -> np.ndarray:
        """Mean stack with fold normalization."""
        # Zero out dead traces
        masked = traces * live_mask

        # Sum and normalize
        summed = np.sum(masked, axis=1)

        with np.errstate(invalid='ignore', divide='ignore'):
            if self.config.normalize:
                stacked = np.where(fold > 0, summed / fold, 0.0)
            else:
                stacked = summed

        return stacked

    def _median_stack(
        self,
        traces: np.ndarray,
        live_mask: np.ndarray,
    ) -> np.ndarray:
        """Median stack (robust to outliers)."""
        n_samples = traces.shape[0]
        stacked = np.zeros(n_samples, dtype=np.float32)

        for i in range(n_samples):
            live_values = traces[i, live_mask[i, :]]
            if len(live_values) > 0:
                stacked[i] = np.median(live_values)

        return stacked

    def _weighted_stack(
        self,
        traces: np.ndarray,
        weights: np.ndarray,
        live_mask: np.ndarray,
    ) -> np.ndarray:
        """Weighted stack with custom weights."""
        # Apply weights and live mask
        weighted = traces * weights * live_mask

        # Sum
        summed = np.sum(weighted, axis=1)

        if self.config.normalize:
            weight_sum = np.sum(weights * live_mask, axis=1)
            with np.errstate(invalid='ignore', divide='ignore'):
                stacked = np.where(weight_sum > 0, summed / weight_sum, 0.0)
        else:
            stacked = summed

        return stacked

    def _compute_weights(
        self,
        traces: np.ndarray,
        offsets: Optional[np.ndarray],
        live_mask: np.ndarray,
    ) -> np.ndarray:
        """Compute stacking weights."""
        n_samples, n_traces = traces.shape

        if self.config.weight_by_offset and offsets is not None:
            # Weight by inverse offset (near offsets weighted more)
            abs_offsets = np.abs(offsets)
            max_offset = np.max(abs_offsets)
            if max_offset > 0:
                # Normalize to [0.5, 1.0] range
                weights = 1.0 - 0.5 * (abs_offsets / max_offset)
            else:
                weights = np.ones(n_traces)
            # Broadcast to all samples
            weights = np.tile(weights, (n_samples, 1))
        else:
            # Equal weights
            weights = np.ones((n_samples, n_traces), dtype=np.float32)

        return weights

    # =========================================================================
    # Batch Stacking
    # =========================================================================

    def stack_line(
        self,
        gathers: List[Tuple[np.ndarray, np.ndarray]],
        sample_interval_ms: float,
        cdp_numbers: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Stack multiple gathers to create a stacked line/section.

        Args:
            gathers: List of (traces, offsets) tuples for each CDP
            sample_interval_ms: Sample interval in milliseconds
            cdp_numbers: Optional list of CDP numbers

        Returns:
            Tuple of (stacked_traces, fold_array, cdp_list)
            where stacked_traces is (n_samples, n_cdps)
        """
        if not gathers:
            raise ValueError("No gathers to stack")

        n_cdps = len(gathers)
        n_samples = gathers[0][0].shape[0]

        stacked = np.zeros((n_samples, n_cdps), dtype=np.float32)
        fold_array = np.zeros((n_samples, n_cdps), dtype=np.int32)

        if cdp_numbers is None:
            cdp_numbers = list(range(n_cdps))

        for i, (traces, offsets) in enumerate(gathers):
            cdp = cdp_numbers[i] if i < len(cdp_numbers) else i

            if self.nmo_processor:
                result = self.stack_gather_with_nmo(
                    traces, offsets, sample_interval_ms, cdp
                )
            else:
                result = self.stack_gather(traces, offsets)

            stacked[:, i] = result.trace
            fold_array[:, i] = result.fold

            self._report_progress(i + 1, n_cdps, f"Stacking CDP {cdp}")

        return stacked, fold_array, cdp_numbers

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for multiprocess transfer."""
        result = {
            'class_name': self.__class__.__name__,
            'module': self.__class__.__module__,
            'params': {
                'config': self.config.to_dict(),
            }
        }

        if self.nmo_processor:
            result['params']['nmo_processor'] = self.nmo_processor.to_dict()

        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CDPStacker':
        """Deserialize from dictionary."""
        from processors.nmo_processor import NMOProcessor

        params = d.get('params', d)
        config = StackConfig.from_dict(params.get('config', {}))

        nmo_processor = None
        if 'nmo_processor' in params:
            nmo_processor = NMOProcessor.from_dict(params['nmo_processor'])

        return cls(config=config, nmo_processor=nmo_processor)


# =============================================================================
# Convenience Functions
# =============================================================================

def stack_cdp_gather(
    traces: np.ndarray,
    method: str = 'mean',
    min_fold: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple CDP stack of a gather.

    Args:
        traces: 2D array (n_samples, n_traces)
        method: 'mean' or 'median'
        min_fold: Minimum fold cutoff

    Returns:
        Tuple of (stacked_trace, fold_array)
    """
    config = StackConfig(method=method, min_fold=min_fold)
    stacker = CDPStacker(config)
    result = stacker.stack_gather(traces)
    return result.trace, result.fold


def nmo_and_stack(
    traces: np.ndarray,
    offsets: np.ndarray,
    velocity_model: VelocityModel,
    sample_interval_ms: float,
    stretch_mute: float = 1.5,
    stack_method: str = 'mean',
    cdp: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combined NMO correction and stacking.

    Args:
        traces: 2D array (n_samples, n_traces)
        offsets: 1D array of offsets
        velocity_model: VelocityModel with RMS velocities
        sample_interval_ms: Sample interval in milliseconds
        stretch_mute: Stretch mute factor
        stack_method: 'mean' or 'median'
        cdp: Optional CDP number

    Returns:
        Tuple of (stacked_trace, fold_array)
    """
    from processors.nmo_processor import NMOProcessor, NMOConfig

    nmo_config = NMOConfig(stretch_mute_factor=stretch_mute)
    nmo = NMOProcessor(nmo_config, velocity_model)

    stack_config = StackConfig(method=stack_method)
    stacker = CDPStacker(stack_config, nmo_processor=nmo)

    result = stacker.stack_gather_with_nmo(traces, offsets, sample_interval_ms, cdp)
    return result.trace, result.fold
