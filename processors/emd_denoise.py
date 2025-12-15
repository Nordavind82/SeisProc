"""
EMD/EEMD Denoising processor using Empirical Mode Decomposition.

Implements adaptive, data-driven signal decomposition into Intrinsic Mode
Functions (IMFs) for non-stationary signal processing.

Methods:
- EMD: Original algorithm (fast, may have mode mixing)
- EEMD: Ensemble EMD with noise-assisted decomposition (robust, slower)
- CEEMDAN: Complete EEMD with Adaptive Noise (best quality, slowest)
- EEMD-Fast: Parallel ensemble EMD (faster than standard EEMD)

Key advantages:
- No predefined basis functions - fully adaptive
- Handles non-linear and non-stationary signals
- Separates signals by instantaneous frequency
- No spectral leakage between modes

Performance optimizations:
- Parallel trace processing with joblib
- Parallel ensemble computation for EEMD-Fast
- Reduced ensemble size option for speed vs quality tradeoff
"""
import numpy as np
from typing import Optional, Literal, List, Union
import logging
import warnings

from models.seismic_data import SeismicData
from processors.base_processor import BaseProcessor

logger = logging.getLogger(__name__)

# Try to import PyEMD
try:
    from PyEMD import EMD, EEMD, CEEMDAN
    PYEMD_AVAILABLE = True
except ImportError:
    PYEMD_AVAILABLE = False
    logger.warning("PyEMD not available. Install with: pip install emd-signal")

# Try to import joblib for parallel processing
try:
    from joblib import Parallel, delayed
    import multiprocessing
    JOBLIB_AVAILABLE = True
    N_JOBS = max(1, multiprocessing.cpu_count() - 1)
except ImportError:
    JOBLIB_AVAILABLE = False
    N_JOBS = 1

# Try to import numba for JIT acceleration
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


class EMDDenoise(BaseProcessor):
    """
    Empirical Mode Decomposition Denoising.

    Decomposes signals into Intrinsic Mode Functions (IMFs) and removes
    selected modes to achieve denoising.

    Best suited for:
    - Non-stationary noise removal
    - Trend/drift removal
    - Ground roll attenuation (often in low-frequency IMFs)
    - Mode separation

    Performance notes:
    - EMD: ~0.1s per trace (fastest, may have mode mixing)
    - EEMD-Fast: ~0.5-2s per trace (parallel ensemble, good quality)
    - EEMD: ~2-10s per trace (sequential ensemble, robust)
    - CEEMDAN: ~5-20s per trace (slowest, best quality)
    """

    def __init__(self,
                 method: Literal['emd', 'eemd', 'eemd_fast', 'ceemdan'] = 'eemd_fast',
                 num_imfs: Optional[int] = None,
                 remove_imfs: Union[str, List[int]] = 'first',
                 ensemble_size: int = 100,
                 noise_amplitude: float = 0.2,
                 spline_kind: str = 'cubic',
                 parallel_ensemble: bool = True,
                 max_siftings: int = 50):
        """
        Initialize EMD-Denoise processor.

        Args:
            method: Decomposition method:
                - 'emd': Original EMD (fast, may have mode mixing)
                - 'eemd_fast': Parallel ensemble EMD (recommended - fast + robust)
                - 'eemd': Standard Ensemble EMD (robust, slower)
                - 'ceemdan': Complete EEMD with Adaptive Noise (best, slowest)
            num_imfs: Maximum number of IMFs to compute (None = automatic)
            remove_imfs: IMFs to remove for denoising:
                - 'first': Remove first IMF (highest frequency noise)
                - 'first_n': Remove first n IMFs (e.g., 'first_2')
                - 'last': Remove last IMF (lowest frequency trend)
                - 'last_n': Remove last n IMFs (e.g., 'last_2')
                - List of indices: Remove specific IMFs [0, 1, 2]
            ensemble_size: Number of ensemble realizations for EEMD/CEEMDAN
                - Recommended: 50-100 for EEMD-Fast, 100-200 for EEMD
            noise_amplitude: Noise amplitude for EEMD/CEEMDAN (as fraction of std)
            spline_kind: Spline interpolation method ('cubic', 'linear', 'quadratic')
            parallel_ensemble: Use parallel processing for ensemble (EEMD-Fast)
            max_siftings: Maximum sifting iterations per IMF (performance tuning)
        """
        if not PYEMD_AVAILABLE:
            raise ImportError("PyEMD required. Install with: pip install emd-signal")

        self.method = method
        self.num_imfs = num_imfs
        self.remove_imfs = remove_imfs
        self.ensemble_size = ensemble_size
        self.noise_amplitude = noise_amplitude
        self.spline_kind = spline_kind
        self.parallel_ensemble = parallel_ensemble
        self.max_siftings = max_siftings

        super().__init__(
            method=method,
            num_imfs=num_imfs,
            remove_imfs=remove_imfs,
            ensemble_size=ensemble_size,
            noise_amplitude=noise_amplitude,
            spline_kind=spline_kind,
            parallel_ensemble=parallel_ensemble,
            max_siftings=max_siftings
        )

    def _validate_params(self):
        """Validate processor parameters."""
        if self.method not in ['emd', 'eemd', 'eemd_fast', 'ceemdan']:
            raise ValueError("method must be 'emd', 'eemd', 'eemd_fast', or 'ceemdan'")
        if self.num_imfs is not None and self.num_imfs < 1:
            raise ValueError("num_imfs must be at least 1")
        if self.ensemble_size < 1:
            raise ValueError("ensemble_size must be at least 1")
        if self.noise_amplitude <= 0:
            raise ValueError("noise_amplitude must be positive")
        if self.spline_kind not in ['cubic', 'linear', 'quadratic']:
            raise ValueError("spline_kind must be 'cubic', 'linear', or 'quadratic'")

        # Validate remove_imfs
        if isinstance(self.remove_imfs, str):
            valid_patterns = ['first', 'last'] + \
                           [f'first_{i}' for i in range(1, 10)] + \
                           [f'last_{i}' for i in range(1, 10)]
            if self.remove_imfs not in valid_patterns:
                raise ValueError(
                    f"Invalid remove_imfs pattern: {self.remove_imfs}. "
                    "Use 'first', 'last', 'first_n', 'last_n', or a list of indices."
                )
        elif isinstance(self.remove_imfs, list):
            if not all(isinstance(i, int) and i >= 0 for i in self.remove_imfs):
                raise ValueError("remove_imfs list must contain non-negative integers")

    def get_description(self) -> str:
        """Get processor description."""
        method_name = self.method.upper().replace('_', '-')
        if isinstance(self.remove_imfs, str):
            remove_str = self.remove_imfs
        else:
            remove_str = f"IMFs {self.remove_imfs}"
        parallel_str = " (parallel)" if self.method == 'eemd_fast' and self.parallel_ensemble else ""
        return f"EMD-Denoise ({method_name}{parallel_str}): remove={remove_str}, n={self.ensemble_size}"

    def process(self, data: SeismicData) -> SeismicData:
        """
        Apply EMD-domain denoising to seismic data.

        Args:
            data: Input seismic data

        Returns:
            Denoised seismic data
        """
        import time

        start_time = time.time()

        # Convert to float32 for memory efficiency (50% savings)
        traces = data.traces
        if traces.dtype != np.float32:
            traces = traces.astype(np.float32)
        else:
            traces = traces.copy()
        n_samples, n_traces = traces.shape

        # Determine processing strategy
        use_trace_parallel = JOBLIB_AVAILABLE and n_traces > 4
        method_display = self.method.upper().replace('_', '-')

        if self.method == 'eemd_fast':
            parallel_info = f"EEMD-Fast (ensemble={self.ensemble_size}, {N_JOBS} cores)"
        elif use_trace_parallel:
            parallel_info = f"Parallel({N_JOBS} cores)"
        else:
            parallel_info = "Sequential"

        logger.info(
            f"EMD-Denoise ({method_display}): {n_traces} traces × {n_samples} samples | "
            f"Remove: {self.remove_imfs} | {parallel_info}"
        )

        # Process traces based on method
        if self.method == 'eemd_fast':
            # EEMD-Fast: Parallel ensemble computation per trace
            denoised_traces = self._process_eemd_fast(traces)
        else:
            # Standard EMD/EEMD/CEEMDAN
            emd_processor = self._create_emd_processor()

            if use_trace_parallel:
                # For EMD (fast), parallelize across traces
                # For EEMD/CEEMDAN (slow), use fewer parallel jobs to avoid memory issues
                n_jobs = N_JOBS if self.method == 'emd' else min(4, N_JOBS)
                results = Parallel(n_jobs=n_jobs, prefer="threads")(
                    delayed(self._process_trace)(traces[:, i], emd_processor)
                    for i in range(n_traces)
                )
                denoised_traces = np.column_stack(results)
            else:
                denoised_traces = np.zeros_like(traces)
                for i in range(n_traces):
                    denoised_traces[:, i] = self._process_trace(traces[:, i], emd_processor)

        # Compute metrics
        elapsed = time.time() - start_time
        throughput = n_traces / elapsed if elapsed > 0 else 0

        input_rms = np.sqrt(np.mean(traces**2))
        output_rms = np.sqrt(np.mean(denoised_traces**2))
        energy_ratio = output_rms / input_rms if input_rms > 0 else 0

        logger.info(
            f"EMD-Denoise complete: {elapsed:.2f}s | "
            f"{throughput:.1f} traces/s | "
            f"Energy: {energy_ratio:.1%} retained"
        )

        return SeismicData(
            traces=denoised_traces,
            sample_rate=data.sample_rate,
            metadata={
                **data.metadata,
                'processor': self.get_description()
            }
        )

    def _process_eemd_fast(self, traces: np.ndarray) -> np.ndarray:
        """
        Process traces using parallel EEMD (EEMD-Fast).

        This method computes ensemble EMD by:
        1. Adding noise realizations in parallel
        2. Computing EMD for each noise realization in parallel
        3. Averaging the IMFs across ensemble

        This is significantly faster than standard EEMD which processes
        ensemble trials sequentially.

        Args:
            traces: Input traces (n_samples, n_traces)

        Returns:
            Denoised traces (n_samples, n_traces)
        """
        n_samples, n_traces = traces.shape
        denoised_traces = np.zeros_like(traces)

        # Process each trace
        for trace_idx in range(n_traces):
            trace = traces[:, trace_idx]
            denoised_traces[:, trace_idx] = self._eemd_fast_single(trace)

        return denoised_traces

    def _eemd_fast_single(self, trace: np.ndarray) -> np.ndarray:
        """
        EEMD-Fast for a single trace with parallel ensemble processing.

        Args:
            trace: 1D signal array

        Returns:
            Denoised trace
        """
        try:
            n_samples = len(trace)
            signal_std = np.std(trace)

            if signal_std < 1e-10:
                return trace

            # Generate noise realizations
            np.random.seed(None)  # Ensure different random seeds
            noise_amplitude = self.noise_amplitude * signal_std

            # Create base EMD processor (optimized settings)
            def run_single_ensemble(seed):
                """Run EMD on signal + noise for one ensemble member."""
                rng = np.random.RandomState(seed)
                noise = rng.randn(n_samples) * noise_amplitude
                noisy_signal = trace + noise

                # Create fresh EMD for this ensemble member
                emd = EMD(spline_kind=self.spline_kind)
                emd.MAX_ITERATION = self.max_siftings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        imfs = emd.emd(noisy_signal)
                        if imfs is None or len(imfs) == 0:
                            return None
                        return imfs
                    except Exception:
                        return None

            # Run ensemble in parallel
            if JOBLIB_AVAILABLE and self.parallel_ensemble:
                seeds = np.random.randint(0, 2**31, self.ensemble_size)
                results = Parallel(n_jobs=N_JOBS, prefer="threads")(
                    delayed(run_single_ensemble)(seed) for seed in seeds
                )
            else:
                # Sequential fallback
                results = []
                for i in range(self.ensemble_size):
                    seed = np.random.randint(0, 2**31)
                    results.append(run_single_ensemble(seed))

            # Filter out failed decompositions
            valid_results = [r for r in results if r is not None]

            if len(valid_results) == 0:
                logger.warning("All EEMD ensemble members failed, returning original")
                return trace

            # Average IMFs across ensemble (handling different IMF counts)
            # Find max number of IMFs
            max_imfs = max(len(imfs) for imfs in valid_results)

            # Pad and average
            averaged_imfs = []
            for imf_idx in range(max_imfs):
                imf_sum = np.zeros(n_samples, dtype=np.float32)
                count = 0
                for imfs in valid_results:
                    if imf_idx < len(imfs):
                        imf = imfs[imf_idx]
                        if len(imf) == n_samples:
                            imf_sum += imf
                            count += 1
                if count > 0:
                    averaged_imfs.append(imf_sum / count)

            if len(averaged_imfs) == 0:
                return trace

            averaged_imfs = np.array(averaged_imfs)

            # Apply IMF removal
            n_imfs = len(averaged_imfs)
            remove_indices = self._get_remove_indices(n_imfs)
            keep_indices = [i for i in range(n_imfs) if i not in remove_indices]

            if len(keep_indices) == 0:
                return np.zeros_like(trace)

            # Reconstruct from kept IMFs
            denoised = np.sum(averaged_imfs[keep_indices], axis=0)
            return denoised

        except Exception as e:
            logger.warning(f"EEMD-Fast failed: {e}. Returning original.")
            return trace

    def _create_emd_processor(self):
        """Create the appropriate EMD processor based on method."""
        if self.method == 'emd':
            processor = EMD(spline_kind=self.spline_kind)
            processor.MAX_ITERATION = self.max_siftings
            if self.num_imfs is not None:
                processor.MAX_ITERATION = min(self.max_siftings, self.num_imfs * 10)
        elif self.method in ['eemd', 'eemd_fast']:
            processor = EEMD(
                trials=self.ensemble_size,
                noise_width=self.noise_amplitude,
                spline_kind=self.spline_kind
            )
        else:  # ceemdan
            processor = CEEMDAN(
                trials=self.ensemble_size,
                epsilon=self.noise_amplitude,
                spline_kind=self.spline_kind
            )

        return processor

    def _process_trace(self, trace: np.ndarray, emd_processor) -> np.ndarray:
        """
        Process a single trace using EMD decomposition.

        Args:
            trace: 1D signal array
            emd_processor: EMD/EEMD/CEEMDAN processor

        Returns:
            Denoised trace
        """
        try:
            # Perform decomposition
            if self.method == 'emd':
                imfs = emd_processor.emd(trace)
            else:
                imfs = emd_processor(trace)

            if imfs is None or len(imfs) == 0:
                return trace

            # IMFs are organized from highest to lowest frequency
            # imfs[0] = highest frequency (often noise)
            # imfs[-1] = residual/trend
            n_imfs = len(imfs)

            # Determine which IMFs to remove
            remove_indices = self._get_remove_indices(n_imfs)

            # Keep all IMFs except those to be removed
            keep_indices = [i for i in range(n_imfs) if i not in remove_indices]

            if len(keep_indices) == 0:
                # All IMFs removed - return zeros
                return np.zeros_like(trace)

            # Reconstruct from kept IMFs
            denoised = np.sum(imfs[keep_indices], axis=0)

            return denoised

        except Exception as e:
            logger.warning(f"EMD failed for trace: {e}. Returning original.")
            return trace

    def _get_remove_indices(self, n_imfs: int) -> List[int]:
        """
        Get list of IMF indices to remove.

        Args:
            n_imfs: Total number of IMFs

        Returns:
            List of indices to remove
        """
        if isinstance(self.remove_imfs, list):
            return [i for i in self.remove_imfs if i < n_imfs]

        if self.remove_imfs == 'first':
            return [0]
        elif self.remove_imfs == 'last':
            return [n_imfs - 1]
        elif self.remove_imfs.startswith('first_'):
            n = int(self.remove_imfs.split('_')[1])
            return list(range(min(n, n_imfs)))
        elif self.remove_imfs.startswith('last_'):
            n = int(self.remove_imfs.split('_')[1])
            return list(range(max(0, n_imfs - n), n_imfs))

        return [0]  # Default: remove first IMF


def get_imfs(trace: np.ndarray, method: str = 'eemd', **kwargs) -> np.ndarray:
    """
    Utility function to get IMFs for a single trace.

    Useful for visualization and analysis.

    Args:
        trace: 1D signal array
        method: 'emd', 'eemd', or 'ceemdan'
        **kwargs: Additional arguments for EMD processor

    Returns:
        Array of IMFs (n_imfs, n_samples)
    """
    if not PYEMD_AVAILABLE:
        raise ImportError("PyEMD required. Install with: pip install emd-signal")

    if method == 'emd':
        processor = EMD(**kwargs)
        return processor.emd(trace)
    elif method == 'eemd':
        processor = EEMD(**kwargs)
        return processor(trace)
    else:
        processor = CEEMDAN(**kwargs)
        return processor(trace)
