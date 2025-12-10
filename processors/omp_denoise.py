"""
Orthogonal Matching Pursuit (OMP) Denoising processor.

Implements sparse representation-based denoising using OMP algorithm
with various dictionary options optimized for seismic data.

Key advantages:
- Sparse signal representation for effective noise separation
- Flexible dictionary selection (DCT, DFT, Gabor, Wavelet, learned)
- Adaptive sparsity control via residual threshold or atom count
- Efficient batch processing with Numba acceleration
- **Robust spatial noise estimation** using neighboring traces (MAD-based)
- **Adaptive sparsity** based on local SNR

Best suited for:
- Signals with sparse representation in chosen dictionary
- Random noise removal while preserving sharp features
- Data with known sparse structure (reflections, wavelets)
- When traditional filtering causes too much signal distortion
- Seismic data with spatially-varying noise levels

Algorithm:
1. Estimate noise level from neighboring traces using MAD (robust to outliers)
2. Compute local SNR and adapt sparsity/tolerance accordingly
3. Divide signal into overlapping patches
4. For each patch, find sparse representation using OMP:
   - Iteratively select dictionary atoms that best explain residual
   - Update coefficients via least squares
   - Stop when residual is small enough or max atoms reached
5. Reconstruct denoised signal from sparse coefficients
6. Average overlapping patches for final output
"""
import numpy as np
from typing import Optional, Literal, Union, Tuple
import logging

from models.seismic_data import SeismicData
from processors.base_processor import BaseProcessor

logger = logging.getLogger(__name__)

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

# Try to import joblib for parallel processing
try:
    from joblib import Parallel, delayed
    import multiprocessing
    JOBLIB_AVAILABLE = True
    N_JOBS = max(1, multiprocessing.cpu_count() - 1)
except ImportError:
    JOBLIB_AVAILABLE = False
    N_JOBS = 1

# Try to import scipy for optimized operations
try:
    from scipy import linalg
    from scipy.fft import dct, idct
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Try to import sklearn for comparison/validation
try:
    from sklearn.linear_model import OrthogonalMatchingPursuit
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# =============================================================================
# Dictionary Generation Functions
# =============================================================================

def create_dct_dictionary(patch_size: int, n_atoms: Optional[int] = None) -> np.ndarray:
    """
    Create overcomplete DCT (Discrete Cosine Transform) dictionary.

    DCT is excellent for smooth signals and compresses energy well.

    Args:
        patch_size: Size of signal patches
        n_atoms: Number of dictionary atoms (default: 2 * patch_size for overcomplete)

    Returns:
        Dictionary matrix (patch_size x n_atoms), columns are normalized atoms
    """
    if n_atoms is None:
        n_atoms = 2 * patch_size

    # Create DCT-II basis (most common, used in JPEG)
    dictionary = np.zeros((patch_size, n_atoms))

    for k in range(n_atoms):
        # DCT basis function with varying frequency
        freq = k * np.pi / n_atoms
        n = np.arange(patch_size)
        atom = np.cos(freq * (n + 0.5))

        # Normalize
        norm = np.linalg.norm(atom)
        if norm > 1e-10:
            dictionary[:, k] = atom / norm
        else:
            dictionary[:, k] = atom

    return dictionary


def create_dft_dictionary(patch_size: int, n_atoms: Optional[int] = None) -> np.ndarray:
    """
    Create overcomplete DFT (Discrete Fourier Transform) dictionary.

    DFT captures both sine and cosine components, good for oscillatory signals.

    Args:
        patch_size: Size of signal patches
        n_atoms: Number of dictionary atoms (default: 2 * patch_size)

    Returns:
        Dictionary matrix (patch_size x n_atoms), real-valued normalized atoms
    """
    if n_atoms is None:
        n_atoms = 2 * patch_size

    atoms = []
    n = np.arange(patch_size)

    # Generate frequencies - more than needed to fill n_atoms
    max_freq_idx = n_atoms  # Generate enough frequencies

    for k in range(max_freq_idx):
        if len(atoms) >= n_atoms:
            break

        freq = 2 * np.pi * k / (2 * patch_size)  # Overcomplete: 2x frequency resolution

        # Cosine component
        atom_cos = np.cos(freq * n)
        norm = np.linalg.norm(atom_cos)
        if norm > 1e-10:
            atoms.append(atom_cos / norm)

        if len(atoms) >= n_atoms:
            break

        # Sine component (skip DC which has zero sine)
        if k > 0:
            atom_sin = np.sin(freq * n)
            norm = np.linalg.norm(atom_sin)
            if norm > 1e-10:
                atoms.append(atom_sin / norm)

    # Ensure we have exactly n_atoms columns
    while len(atoms) < n_atoms:
        # Fill with additional DCT atoms if needed
        k = len(atoms)
        freq = k * np.pi / n_atoms
        atom = np.cos(freq * (n + 0.5))
        norm = np.linalg.norm(atom)
        atoms.append(atom / norm if norm > 1e-10 else atom)

    dictionary = np.column_stack(atoms[:n_atoms])
    return dictionary


def create_gabor_dictionary(patch_size: int, n_atoms: Optional[int] = None,
                           n_scales: int = 4, n_frequencies: int = 8) -> np.ndarray:
    """
    Create Gabor dictionary with varying scales and frequencies.

    Gabor atoms are localized in both time and frequency, ideal for
    non-stationary signals like seismic wavelets.

    Args:
        patch_size: Size of signal patches
        n_atoms: Target number of atoms (approximate)
        n_scales: Number of Gaussian envelope scales
        n_frequencies: Number of modulation frequencies per scale

    Returns:
        Dictionary matrix (patch_size x n_atoms), normalized atoms
    """
    if n_atoms is None:
        n_atoms = 2 * patch_size

    atoms = []
    t = np.linspace(-1, 1, patch_size)

    # Generate scales (standard deviations of Gaussian envelope)
    min_scale = 0.05
    max_scale = 0.5
    scales = np.logspace(np.log10(min_scale), np.log10(max_scale), n_scales)

    # Generate frequencies
    max_freq = patch_size / 4
    frequencies = np.linspace(0, max_freq, n_frequencies)

    # Generate positions (shifts)
    n_positions = max(1, n_atoms // (n_scales * n_frequencies * 2))
    positions = np.linspace(-0.5, 0.5, n_positions)

    for sigma in scales:
        for freq in frequencies:
            for pos in positions:
                # Gaussian envelope
                envelope = np.exp(-((t - pos) ** 2) / (2 * sigma ** 2))

                # Cosine modulation
                atom_cos = envelope * np.cos(2 * np.pi * freq * t)
                norm = np.linalg.norm(atom_cos)
                if norm > 1e-10:
                    atoms.append(atom_cos / norm)

                # Sine modulation (skip DC)
                if freq > 0:
                    atom_sin = envelope * np.sin(2 * np.pi * freq * t)
                    norm = np.linalg.norm(atom_sin)
                    if norm > 1e-10:
                        atoms.append(atom_sin / norm)

                if len(atoms) >= n_atoms:
                    break
            if len(atoms) >= n_atoms:
                break
        if len(atoms) >= n_atoms:
            break

    dictionary = np.column_stack(atoms[:n_atoms])
    return dictionary


def create_wavelet_dictionary(patch_size: int, n_atoms: Optional[int] = None,
                             wavelet_type: str = 'ricker') -> np.ndarray:
    """
    Create wavelet dictionary with varying scales and positions.

    Wavelets are excellent for seismic data which contains wavelet-like reflections.

    Args:
        patch_size: Size of signal patches
        n_atoms: Target number of atoms
        wavelet_type: Type of wavelet ('ricker', 'morlet', 'gaussian_derivative')

    Returns:
        Dictionary matrix (patch_size x n_atoms), normalized atoms
    """
    if n_atoms is None:
        n_atoms = 2 * patch_size

    atoms = []
    t = np.linspace(-1, 1, patch_size)

    # Generate scales
    n_scales = int(np.sqrt(n_atoms))
    min_scale = 0.02
    max_scale = 0.3
    scales = np.logspace(np.log10(min_scale), np.log10(max_scale), n_scales)

    # Generate positions
    n_positions = n_atoms // n_scales + 1
    positions = np.linspace(-0.8, 0.8, n_positions)

    for scale in scales:
        for pos in positions:
            t_shifted = (t - pos) / scale

            if wavelet_type == 'ricker':
                # Ricker (Mexican hat) wavelet - common in seismic
                atom = (1 - 2 * np.pi**2 * t_shifted**2) * np.exp(-np.pi**2 * t_shifted**2)
            elif wavelet_type == 'morlet':
                # Morlet wavelet
                sigma = 1.0
                atom = np.exp(-t_shifted**2 / (2 * sigma**2)) * np.cos(5 * t_shifted)
            else:  # gaussian_derivative
                # First derivative of Gaussian
                atom = -t_shifted * np.exp(-t_shifted**2 / 2)

            norm = np.linalg.norm(atom)
            if norm > 1e-10:
                atoms.append(atom / norm)

            if len(atoms) >= n_atoms:
                break
        if len(atoms) >= n_atoms:
            break

    # Pad with DCT atoms if needed
    while len(atoms) < n_atoms:
        k = len(atoms)
        freq = k * np.pi / n_atoms
        n = np.arange(patch_size)
        atom = np.cos(freq * (n + 0.5))
        norm = np.linalg.norm(atom)
        if norm > 1e-10:
            atoms.append(atom / norm)
        else:
            atoms.append(atom)

    dictionary = np.column_stack(atoms[:n_atoms])
    return dictionary


def create_hybrid_dictionary(patch_size: int, n_atoms: Optional[int] = None) -> np.ndarray:
    """
    Create hybrid dictionary combining DCT, wavelets, and Gabor atoms.

    This provides a diverse set of atoms suitable for various signal components.

    Args:
        patch_size: Size of signal patches
        n_atoms: Total number of atoms

    Returns:
        Dictionary matrix (patch_size x n_atoms), normalized atoms
    """
    if n_atoms is None:
        n_atoms = 3 * patch_size

    # Allocate atoms to different bases
    n_dct = n_atoms // 3
    n_wavelet = n_atoms // 3
    n_gabor = n_atoms - n_dct - n_wavelet

    dct_dict = create_dct_dictionary(patch_size, n_dct)
    wavelet_dict = create_wavelet_dictionary(patch_size, n_wavelet, 'ricker')
    gabor_dict = create_gabor_dictionary(patch_size, n_gabor)

    dictionary = np.hstack([dct_dict, wavelet_dict, gabor_dict])

    # Remove any duplicate or near-duplicate atoms
    # (correlation > 0.99)
    keep_mask = np.ones(dictionary.shape[1], dtype=bool)
    for i in range(dictionary.shape[1]):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, dictionary.shape[1]):
            if keep_mask[j]:
                corr = np.abs(np.dot(dictionary[:, i], dictionary[:, j]))
                if corr > 0.99:
                    keep_mask[j] = False

    dictionary = dictionary[:, keep_mask]

    # Pad if we removed too many
    while dictionary.shape[1] < n_atoms:
        k = dictionary.shape[1]
        freq = k * np.pi / n_atoms
        n = np.arange(patch_size)
        atom = np.cos(freq * (n + 0.5))
        norm = np.linalg.norm(atom)
        atom = atom / norm if norm > 1e-10 else atom
        dictionary = np.column_stack([dictionary, atom])

    return dictionary[:, :n_atoms]


# =============================================================================
# Robust Spatial Noise Estimation
# =============================================================================

def estimate_noise_mad(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Estimate noise standard deviation using Median Absolute Deviation (MAD).

    MAD is robust to outliers (signal spikes) unlike standard deviation.
    For Gaussian noise: sigma ≈ MAD / 0.6745

    Args:
        data: Input data array
        axis: Axis along which to compute MAD

    Returns:
        Estimated noise standard deviation
    """
    median = np.median(data, axis=axis, keepdims=True)
    mad = np.median(np.abs(data - median), axis=axis)
    # Scale factor for Gaussian: MAD = 0.6745 * sigma
    return mad / 0.6745


def estimate_noise_wavelet(trace: np.ndarray) -> float:
    """
    Estimate noise using finest wavelet detail coefficients (Donoho method).

    Uses the high-frequency wavelet coefficients which are dominated by noise.

    Args:
        trace: 1D signal

    Returns:
        Estimated noise standard deviation
    """
    # Use Haar wavelet (simplest) - detail coefficients at finest scale
    n = len(trace)
    if n < 4:
        return estimate_noise_mad(trace)

    # Haar detail coefficients: d[k] = (x[2k] - x[2k+1]) / sqrt(2)
    n_pairs = n // 2
    detail = (trace[:2*n_pairs:2] - trace[1:2*n_pairs:2]) / np.sqrt(2)

    # MAD of detail coefficients
    return estimate_noise_mad(detail)


def estimate_noise_spatial(traces: np.ndarray, trace_idx: int, aperture: int,
                          method: str = 'mad_diff') -> float:
    """
    Estimate noise level using neighboring traces (spatial redundancy).

    Uses the assumption that signal is coherent across traces while noise is not.
    The difference between adjacent traces isolates the incoherent (noise) component.

    Args:
        traces: Full trace array (n_samples x n_traces)
        trace_idx: Index of center trace
        aperture: Number of neighboring traces to use
        method: Estimation method:
            - 'mad_diff': MAD of differences between adjacent traces
            - 'mad_residual': MAD of residual after local mean subtraction
            - 'wavelet': Wavelet-based estimation on residual

    Returns:
        Estimated noise standard deviation for the center trace
    """
    n_samples, n_traces = traces.shape
    half_ap = aperture // 2

    # Get aperture window
    start = max(0, trace_idx - half_ap)
    end = min(n_traces, trace_idx + half_ap + 1)
    local_traces = traces[:, start:end]
    n_local = local_traces.shape[1]

    if n_local < 2:
        # Fall back to single-trace estimation
        return estimate_noise_wavelet(traces[:, trace_idx])

    if method == 'mad_diff':
        # Difference between adjacent traces isolates incoherent noise
        # For zero-mean uncorrelated noise: var(diff) = 2 * var(noise)
        diffs = np.diff(local_traces, axis=1)
        noise_std = estimate_noise_mad(diffs.flatten()) / np.sqrt(2)

    elif method == 'mad_residual':
        # Subtract local mean (coherent signal estimate)
        local_mean = np.mean(local_traces, axis=1, keepdims=True)
        residual = local_traces - local_mean
        # Residual contains noise + small signal variations
        noise_std = estimate_noise_mad(residual.flatten())

    elif method == 'wavelet':
        # Wavelet estimation on the residual
        local_mean = np.median(local_traces, axis=1)  # More robust than mean
        residual = traces[:, trace_idx] - local_mean
        noise_std = estimate_noise_wavelet(residual)

    else:
        noise_std = estimate_noise_mad(traces[:, trace_idx])

    return max(noise_std, 1e-10)  # Avoid zero


def estimate_local_snr(traces: np.ndarray, trace_idx: int, aperture: int,
                      time_window: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate local SNR along the trace using spatial statistics.

    Computes time-varying SNR by comparing signal power (coherent across traces)
    to noise power (incoherent).

    Args:
        traces: Full trace array (n_samples x n_traces)
        trace_idx: Index of center trace
        aperture: Number of neighboring traces
        time_window: Window size for local estimation (None = use patch_size/4)

    Returns:
        signal_estimate: Estimated clean signal
        snr_db: Local SNR in dB (per sample or per window)
    """
    n_samples, n_traces = traces.shape
    half_ap = aperture // 2

    if time_window is None:
        time_window = max(16, n_samples // 16)

    # Get aperture window
    start = max(0, trace_idx - half_ap)
    end = min(n_traces, trace_idx + half_ap + 1)
    local_traces = traces[:, start:end]

    # Signal estimate: median across traces (robust to outliers)
    signal_estimate = np.median(local_traces, axis=1)

    # Noise estimate: MAD of residuals
    center_in_local = trace_idx - start
    residual = local_traces[:, center_in_local] - signal_estimate

    # Compute windowed SNR
    n_windows = n_samples // time_window
    snr_db = np.zeros(n_windows)

    for i in range(n_windows):
        ws = i * time_window
        we = min(ws + time_window, n_samples)

        signal_power = np.mean(signal_estimate[ws:we] ** 2)
        noise_power = np.mean(residual[ws:we] ** 2)

        if noise_power > 1e-20:
            snr_db[i] = 10 * np.log10(signal_power / noise_power + 1e-10)
        else:
            snr_db[i] = 40.0  # Cap at 40 dB

    return signal_estimate, snr_db


def compute_adaptive_sparsity(snr_db: float, base_sparsity: int,
                             min_sparsity: int = 2, max_sparsity: int = 30,
                             snr_low: float = 0.0, snr_high: float = 20.0) -> int:
    """
    Compute adaptive sparsity based on local SNR.

    Low SNR → fewer atoms (more aggressive denoising)
    High SNR → more atoms (preserve detail)

    Args:
        snr_db: Local SNR in dB
        base_sparsity: Base sparsity level
        min_sparsity: Minimum sparsity (for low SNR)
        max_sparsity: Maximum sparsity (for high SNR)
        snr_low: SNR threshold below which min_sparsity is used
        snr_high: SNR threshold above which max_sparsity is used

    Returns:
        Adapted sparsity level
    """
    if snr_db <= snr_low:
        return min_sparsity
    elif snr_db >= snr_high:
        return max_sparsity
    else:
        # Linear interpolation
        t = (snr_db - snr_low) / (snr_high - snr_low)
        return int(min_sparsity + t * (max_sparsity - min_sparsity))


def compute_adaptive_tolerance(noise_std: float, signal_std: float,
                              base_tol: float = 0.1) -> float:
    """
    Compute adaptive residual tolerance based on noise level.

    Tolerance is set relative to estimated noise level.

    Args:
        noise_std: Estimated noise standard deviation
        signal_std: Signal standard deviation
        base_tol: Base tolerance factor

    Returns:
        Adapted tolerance (as fraction of signal norm)
    """
    if signal_std < 1e-10:
        return base_tol

    # Noise-to-signal ratio
    nsr = noise_std / signal_std

    # Tolerance should be proportional to noise level
    # Higher noise → stop earlier (larger tolerance)
    adapted_tol = base_tol * (1 + nsr)

    # Clamp to reasonable range
    return np.clip(adapted_tol, 0.01, 0.5)


# =============================================================================
# OMP Algorithm Implementation
# =============================================================================

def omp_cholesky(dictionary: np.ndarray, signal: np.ndarray,
                n_nonzero: int, tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Orthogonal Matching Pursuit using Cholesky decomposition for efficiency.

    This implementation uses incremental Cholesky updates which is O(k^2) per
    iteration instead of O(k^3) for full least squares.

    Args:
        dictionary: Dictionary matrix (n_features x n_atoms)
        signal: Signal to decompose (n_features,)
        n_nonzero: Maximum number of non-zero coefficients
        tol: Residual tolerance for early stopping

    Returns:
        coefficients: Sparse coefficient vector (n_atoms,)
        support: Indices of selected atoms
    """
    n_features, n_atoms = dictionary.shape

    # Pre-compute dictionary correlations for efficiency
    # Gram matrix G = D^T D (computed on-the-fly for selected atoms)
    DtD = dictionary.T @ dictionary  # n_atoms x n_atoms
    Dty = dictionary.T @ signal      # n_atoms

    # Initialize
    residual = signal.copy()
    support = []
    coefficients = np.zeros(n_atoms)

    # Cholesky factor (incrementally updated)
    L = np.zeros((n_nonzero, n_nonzero))

    residual_norm_sq = np.dot(residual, residual)
    signal_norm_sq = residual_norm_sq

    for k in range(n_nonzero):
        # Find atom with maximum correlation to residual
        correlations = dictionary.T @ residual
        correlations[support] = 0  # Exclude already selected atoms

        best_atom = np.argmax(np.abs(correlations))

        # Check if correlation is too small (convergence)
        if np.abs(correlations[best_atom]) < tol * np.sqrt(signal_norm_sq):
            break

        support.append(best_atom)

        # Update Cholesky factor incrementally
        if k == 0:
            L[0, 0] = 1.0
        else:
            # Solve L * w = G[support[:-1], best_atom]
            w = np.zeros(k)
            g = DtD[support[:-1], best_atom]

            # Forward substitution
            for i in range(k):
                w[i] = g[i]
                for j in range(i):
                    w[i] -= L[i, j] * w[j]
                w[i] /= L[i, i]

            L[k, :k] = w
            L[k, k] = np.sqrt(max(1.0 - np.dot(w, w), 1e-10))

        # Solve for coefficients using Cholesky: L L^T x = D_S^T y
        # Forward substitution: L z = Dty[support]
        z = np.zeros(k + 1)
        rhs = Dty[support]
        for i in range(k + 1):
            z[i] = rhs[i]
            for j in range(i):
                z[i] -= L[i, j] * z[j]
            z[i] /= L[i, i]

        # Back substitution: L^T x = z
        x = np.zeros(k + 1)
        for i in range(k, -1, -1):
            x[i] = z[i]
            for j in range(i + 1, k + 1):
                x[i] -= L[j, i] * x[j]
            x[i] /= L[i, i]

        # Update residual
        coefficients[support] = x
        residual = signal - dictionary[:, support] @ x

        residual_norm_sq = np.dot(residual, residual)

        # Check residual tolerance
        if residual_norm_sq < tol * tol * signal_norm_sq:
            break

    return coefficients, np.array(support)


def omp_batch(dictionary: np.ndarray, signals: np.ndarray,
              n_nonzero: int, tol: float = 1e-6) -> np.ndarray:
    """
    Batch OMP for processing multiple signals efficiently.

    Args:
        dictionary: Dictionary matrix (n_features x n_atoms)
        signals: Signal matrix (n_features x n_signals)
        n_nonzero: Maximum number of non-zero coefficients per signal
        tol: Residual tolerance

    Returns:
        coefficients: Sparse coefficient matrix (n_atoms x n_signals)
    """
    n_features, n_atoms = dictionary.shape
    n_signals = signals.shape[1]

    coefficients = np.zeros((n_atoms, n_signals))

    for i in range(n_signals):
        coef, _ = omp_cholesky(dictionary, signals[:, i], n_nonzero, tol)
        coefficients[:, i] = coef

    return coefficients


if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def _omp_core_numba(DtD, Dty, dictionary, signal, n_nonzero, tol):
        """Numba-accelerated OMP core loop."""
        n_features, n_atoms = dictionary.shape

        residual = signal.copy()
        support = np.zeros(n_nonzero, dtype=np.int64)
        n_support = 0
        coefficients = np.zeros(n_atoms)

        L = np.zeros((n_nonzero, n_nonzero))

        signal_norm_sq = np.dot(signal, signal)

        for k in range(n_nonzero):
            # Find best atom
            correlations = dictionary.T @ residual
            for s in range(n_support):
                correlations[support[s]] = 0.0

            best_atom = 0
            best_corr = 0.0
            for j in range(n_atoms):
                if np.abs(correlations[j]) > best_corr:
                    best_corr = np.abs(correlations[j])
                    best_atom = j

            if best_corr < tol * np.sqrt(signal_norm_sq):
                break

            support[n_support] = best_atom
            n_support += 1

            # Update Cholesky
            if k == 0:
                L[0, 0] = 1.0
            else:
                w = np.zeros(k)
                for i in range(k):
                    g_i = DtD[support[i], best_atom]
                    w[i] = g_i
                    for j in range(i):
                        w[i] -= L[i, j] * w[j]
                    w[i] /= L[i, i]

                L[k, :k] = w
                L[k, k] = np.sqrt(max(1.0 - np.dot(w, w), 1e-10))

            # Solve for coefficients
            z = np.zeros(k + 1)
            for i in range(k + 1):
                z[i] = Dty[support[i]]
                for j in range(i):
                    z[i] -= L[i, j] * z[j]
                z[i] /= L[i, i]

            x = np.zeros(k + 1)
            for i in range(k, -1, -1):
                x[i] = z[i]
                for j in range(i + 1, k + 1):
                    x[i] -= L[j, i] * x[j]
                x[i] /= L[i, i]

            # Update coefficients and residual
            for i in range(k + 1):
                coefficients[support[i]] = x[i]

            residual = signal.copy()
            for i in range(k + 1):
                residual -= x[i] * dictionary[:, support[i]]

            residual_norm_sq = np.dot(residual, residual)
            if residual_norm_sq < tol * tol * signal_norm_sq:
                break

        return coefficients, support[:n_support]


def omp_fast(dictionary: np.ndarray, signal: np.ndarray,
             n_nonzero: int, tol: float = 1e-6,
             precomputed_gram: Optional[np.ndarray] = None,
             precomputed_Dty: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast OMP with optional precomputed matrices and Numba acceleration.

    Args:
        dictionary: Dictionary matrix (n_features x n_atoms)
        signal: Signal to decompose (n_features,)
        n_nonzero: Maximum number of non-zero coefficients
        tol: Residual tolerance
        precomputed_gram: Precomputed D^T D matrix (optional)
        precomputed_Dty: Precomputed D^T y vector (optional)

    Returns:
        coefficients: Sparse coefficient vector
        support: Indices of selected atoms
    """
    DtD = precomputed_gram if precomputed_gram is not None else dictionary.T @ dictionary
    Dty = precomputed_Dty if precomputed_Dty is not None else dictionary.T @ signal

    if NUMBA_AVAILABLE:
        return _omp_core_numba(DtD, Dty, dictionary, signal, n_nonzero, tol)
    else:
        return omp_cholesky(dictionary, signal, n_nonzero, tol)


# =============================================================================
# OMP Denoise Processor
# =============================================================================

class OMPDenoise(BaseProcessor):
    """
    Orthogonal Matching Pursuit (OMP) based denoising.

    Uses sparse representation in an overcomplete dictionary to separate
    signal from noise. The signal is assumed to have a sparse representation
    while noise does not.

    Processing steps:
    1. Estimate noise level from neighboring traces (robust MAD-based)
    2. Compute local SNR and adapt sparsity/tolerance accordingly
    3. Extract overlapping patches from each trace
    4. Find sparse representation of each patch using OMP
    5. Reconstruct patches from sparse coefficients
    6. Average overlapping regions for final output

    Supports spatial aperture processing for improved denoising using
    neighboring trace information and robust noise estimation.
    """

    def __init__(self,
                 patch_size: int = 64,
                 overlap: float = 0.5,
                 n_atoms: Optional[int] = None,
                 sparsity: Union[int, float] = 0.1,
                 residual_tol: float = 0.1,
                 dictionary_type: Literal['dct', 'dft', 'gabor', 'wavelet', 'hybrid'] = 'dct',
                 wavelet_type: str = 'ricker',
                 aperture: int = 1,
                 denoise_mode: Literal['patch', 'spatial', 'adaptive'] = 'patch',
                 noise_estimation: Literal['none', 'mad_diff', 'mad_residual', 'wavelet'] = 'mad_diff',
                 adaptive_sparsity: bool = True,
                 min_sparsity: Optional[int] = None,
                 max_sparsity: Optional[int] = None):
        """
        Initialize OMP-Denoise processor.

        Args:
            patch_size: Size of signal patches for sparse coding
            overlap: Overlap fraction between patches (0 to 0.9)
            n_atoms: Number of dictionary atoms (default: 2 * patch_size)
            sparsity: Base sparsity level - if int, max atoms per patch;
                     if float < 1, fraction of patch_size
            residual_tol: Base residual tolerance as fraction of signal norm
            dictionary_type: Type of dictionary to use:
                - 'dct': Discrete Cosine Transform (fast, good for smooth signals)
                - 'dft': Discrete Fourier Transform (oscillatory signals)
                - 'gabor': Gabor atoms (localized time-frequency)
                - 'wavelet': Wavelet dictionary (seismic wavelets)
                - 'hybrid': Combination of multiple bases
            wavelet_type: Wavelet type if dictionary_type='wavelet'
                ('ricker', 'morlet', 'gaussian_derivative')
            aperture: Spatial aperture for multi-trace processing (1 = single trace)
            denoise_mode: Processing mode:
                - 'patch': Independent patch processing (fastest)
                - 'spatial': Joint sparse coding across aperture
                - 'adaptive': Adaptive sparsity based on local SNR
            noise_estimation: Method for spatial noise estimation:
                - 'none': No spatial noise estimation (use fixed params)
                - 'mad_diff': MAD of trace differences (robust, recommended)
                - 'mad_residual': MAD after local mean subtraction
                - 'wavelet': Wavelet-based estimation on residual
            adaptive_sparsity: If True, adapt sparsity based on local SNR
            min_sparsity: Minimum sparsity for low SNR regions (default: sparsity // 2)
            max_sparsity: Maximum sparsity for high SNR regions (default: sparsity * 2)
        """
        self.patch_size = patch_size
        self.overlap = overlap
        self.n_atoms = n_atoms if n_atoms is not None else 2 * patch_size
        self.residual_tol = residual_tol
        self.dictionary_type = dictionary_type
        self.wavelet_type = wavelet_type
        self.aperture = aperture
        self.denoise_mode = denoise_mode
        self.noise_estimation = noise_estimation
        self.adaptive_sparsity = adaptive_sparsity

        # Convert sparsity to integer (max atoms)
        if isinstance(sparsity, float) and sparsity < 1:
            self.sparsity = max(1, int(sparsity * patch_size))
        else:
            self.sparsity = int(sparsity)

        # Set adaptive sparsity bounds
        self.min_sparsity = min_sparsity if min_sparsity is not None else max(2, self.sparsity // 2)
        self.max_sparsity = max_sparsity if max_sparsity is not None else min(self.n_atoms, self.sparsity * 2)

        # Dictionary will be created on first use
        self._dictionary = None
        self._gram_matrix = None

        super().__init__(
            patch_size=patch_size,
            overlap=overlap,
            n_atoms=self.n_atoms,
            sparsity=self.sparsity,
            residual_tol=residual_tol,
            dictionary_type=dictionary_type,
            wavelet_type=wavelet_type,
            aperture=aperture,
            denoise_mode=denoise_mode,
            noise_estimation=noise_estimation,
            adaptive_sparsity=adaptive_sparsity,
            min_sparsity=self.min_sparsity,
            max_sparsity=self.max_sparsity
        )

    def _validate_params(self):
        """Validate processor parameters."""
        if self.patch_size < 8:
            raise ValueError("patch_size must be at least 8")
        if self.overlap < 0 or self.overlap >= 1:
            raise ValueError("overlap must be in [0, 1)")
        if self.n_atoms < self.patch_size:
            raise ValueError("n_atoms must be at least patch_size")
        if self.sparsity < 1:
            raise ValueError("sparsity must be at least 1")
        if self.sparsity > self.n_atoms:
            raise ValueError("sparsity cannot exceed n_atoms")
        if self.residual_tol <= 0 or self.residual_tol >= 1:
            raise ValueError("residual_tol must be in (0, 1)")
        if self.dictionary_type not in ['dct', 'dft', 'gabor', 'wavelet', 'hybrid']:
            raise ValueError("dictionary_type must be 'dct', 'dft', 'gabor', 'wavelet', or 'hybrid'")
        if self.aperture < 1:
            raise ValueError("aperture must be at least 1")
        if self.aperture % 2 == 0:
            raise ValueError("aperture must be odd")
        if self.denoise_mode not in ['patch', 'spatial', 'adaptive']:
            raise ValueError("denoise_mode must be 'patch', 'spatial', or 'adaptive'")
        if self.noise_estimation not in ['none', 'mad_diff', 'mad_residual', 'wavelet']:
            raise ValueError("noise_estimation must be 'none', 'mad_diff', 'mad_residual', or 'wavelet'")

    def _create_dictionary(self) -> np.ndarray:
        """Create the dictionary based on configuration."""
        if self.dictionary_type == 'dct':
            return create_dct_dictionary(self.patch_size, self.n_atoms)
        elif self.dictionary_type == 'dft':
            return create_dft_dictionary(self.patch_size, self.n_atoms)
        elif self.dictionary_type == 'gabor':
            return create_gabor_dictionary(self.patch_size, self.n_atoms)
        elif self.dictionary_type == 'wavelet':
            return create_wavelet_dictionary(self.patch_size, self.n_atoms, self.wavelet_type)
        else:  # hybrid
            return create_hybrid_dictionary(self.patch_size, self.n_atoms)

    def get_description(self) -> str:
        """Get processor description."""
        return (f"OMP-Denoise: patch={self.patch_size}, "
                f"atoms={self.n_atoms}, sparsity={self.sparsity}, "
                f"dict={self.dictionary_type}, aperture={self.aperture}")

    def process(self, data: SeismicData) -> SeismicData:
        """
        Apply OMP-based denoising to seismic data.

        Args:
            data: Input seismic data

        Returns:
            Denoised seismic data
        """
        import time

        start_time = time.time()

        # Convert to float64 for Numba compatibility and numerical precision
        traces = data.traces.astype(np.float64)
        n_samples, n_traces = traces.shape

        # Create dictionary if not exists
        if self._dictionary is None:
            logger.info(f"Creating {self.dictionary_type.upper()} dictionary: "
                       f"{self.patch_size} x {self.n_atoms}")
            self._dictionary = self._create_dictionary()
            self._gram_matrix = self._dictionary.T @ self._dictionary

        # Estimate noise levels per trace if spatial estimation enabled
        noise_estimates = None
        snr_estimates = None
        if self.noise_estimation != 'none' and self.aperture > 1:
            logger.info(f"Estimating noise using {self.noise_estimation} method (aperture={self.aperture})")
            noise_estimates = np.zeros(n_traces)
            for i in range(n_traces):
                noise_estimates[i] = estimate_noise_spatial(
                    traces, i, self.aperture, method=self.noise_estimation
                )

            # Compute global and per-trace SNR
            signal_std = np.std(traces)
            mean_noise = np.mean(noise_estimates)
            global_snr = 20 * np.log10(signal_std / mean_noise) if mean_noise > 1e-10 else 40
            logger.info(f"Noise estimation: mean σ={mean_noise:.4f}, global SNR≈{global_snr:.1f} dB")

            # Estimate local SNR per trace if adaptive mode
            if self.adaptive_sparsity:
                snr_estimates = np.zeros(n_traces)
                for i in range(n_traces):
                    trace_std = np.std(traces[:, i])
                    snr_estimates[i] = 20 * np.log10(trace_std / noise_estimates[i]) if noise_estimates[i] > 1e-10 else 40

        # Log processing info
        step = int(self.patch_size * (1 - self.overlap))
        n_patches_per_trace = (n_samples - self.patch_size) // step + 1

        parallel_info = f"Parallel({N_JOBS} cores)" if JOBLIB_AVAILABLE else "Sequential"
        numba_info = "Numba" if NUMBA_AVAILABLE else "NumPy"
        noise_info = f"Noise: {self.noise_estimation}" if self.noise_estimation != 'none' else "Fixed"
        adaptive_info = f"Adaptive({self.min_sparsity}-{self.max_sparsity})" if self.adaptive_sparsity else f"Fixed({self.sparsity})"

        logger.info(
            f"OMP-Denoise: {n_traces} traces x {n_samples} samples | "
            f"Patches: {n_patches_per_trace}/trace x {self.patch_size} samples | "
            f"Sparsity: {adaptive_info} | {noise_info} | "
            f"{parallel_info} | {numba_info}"
        )

        # Process based on mode
        if self.denoise_mode == 'spatial' and self.aperture > 1:
            denoised_traces = self._process_spatial(traces)
        elif self.denoise_mode == 'adaptive' and noise_estimates is not None:
            denoised_traces = self._process_adaptive(traces, noise_estimates, snr_estimates)
        else:
            denoised_traces = self._process_patches(traces, noise_estimates, snr_estimates)

        # Compute metrics
        elapsed = time.time() - start_time
        throughput = n_traces / elapsed if elapsed > 0 else 0

        input_rms = np.sqrt(np.mean(traces**2))
        output_rms = np.sqrt(np.mean(denoised_traces**2))
        energy_ratio = output_rms / input_rms if input_rms > 0 else 0

        logger.info(
            f"OMP-Denoise complete: {elapsed:.2f}s | "
            f"{throughput:.1f} traces/s | "
            f"Energy: {energy_ratio:.1%} retained"
        )

        # Convert back to original dtype
        output_traces = denoised_traces.astype(data.traces.dtype)

        return SeismicData(
            traces=output_traces,
            sample_rate=data.sample_rate,
            metadata={
                **data.metadata,
                'processor': self.get_description()
            }
        )

    def _process_patches(self, traces: np.ndarray,
                         noise_estimates: Optional[np.ndarray] = None,
                         snr_estimates: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Process traces independently using patch-based OMP.

        Args:
            traces: Input trace array (n_samples x n_traces)
            noise_estimates: Per-trace noise std estimates (optional)
            snr_estimates: Per-trace SNR in dB (optional)

        Returns:
            Denoised traces
        """
        n_samples, n_traces = traces.shape
        denoised = np.zeros_like(traces)

        step = int(self.patch_size * (1 - self.overlap))

        def process_trace(trace_idx):
            trace = traces[:, trace_idx]
            trace_denoised = np.zeros(n_samples)
            trace_weights = np.zeros(n_samples)

            # Get adaptive parameters for this trace
            if self.adaptive_sparsity and snr_estimates is not None:
                local_snr = snr_estimates[trace_idx]
                local_sparsity = compute_adaptive_sparsity(
                    local_snr, self.sparsity,
                    self.min_sparsity, self.max_sparsity
                )
            else:
                local_sparsity = self.sparsity

            # Adaptive tolerance based on noise estimate
            if noise_estimates is not None:
                trace_std = np.std(trace)
                local_tol = compute_adaptive_tolerance(
                    noise_estimates[trace_idx], trace_std, self.residual_tol
                )
            else:
                local_tol = self.residual_tol

            for start in range(0, n_samples - self.patch_size + 1, step):
                end = start + self.patch_size
                patch = trace[start:end]

                # Normalize patch
                patch_mean = np.mean(patch)
                patch_centered = patch - patch_mean
                patch_std = np.std(patch_centered)
                if patch_std > 1e-10:
                    patch_normalized = patch_centered / patch_std
                else:
                    patch_normalized = patch_centered

                # OMP sparse coding with adaptive parameters
                Dty = self._dictionary.T @ patch_normalized
                coef, _ = omp_fast(
                    self._dictionary, patch_normalized,
                    local_sparsity, local_tol,
                    self._gram_matrix, Dty
                )

                # Reconstruct
                reconstructed = self._dictionary @ coef

                # Denormalize
                if patch_std > 1e-10:
                    reconstructed = reconstructed * patch_std
                reconstructed = reconstructed + patch_mean

                # Accumulate with overlap weighting
                trace_denoised[start:end] += reconstructed
                trace_weights[start:end] += 1.0

            # Normalize by weights
            mask = trace_weights > 0
            trace_denoised[mask] /= trace_weights[mask]

            # Handle edges (samples not covered by any patch)
            if not mask[0]:
                trace_denoised[:step] = trace[:step]
            if not mask[-1]:
                trace_denoised[-step:] = trace[-step:]

            return trace_denoised

        if JOBLIB_AVAILABLE and n_traces > 10:
            results = Parallel(n_jobs=N_JOBS, prefer="threads")(
                delayed(process_trace)(i) for i in range(n_traces)
            )
            denoised = np.column_stack(results)
        else:
            for i in range(n_traces):
                denoised[:, i] = process_trace(i)

        return denoised

    def _process_adaptive(self, traces: np.ndarray,
                          noise_estimates: np.ndarray,
                          snr_estimates: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Process with time-varying adaptive sparsity based on local SNR.

        This mode computes SNR per time window and adapts sparsity locally,
        providing finer control than per-trace adaptation.

        Args:
            traces: Input trace array (n_samples x n_traces)
            noise_estimates: Per-trace noise std estimates
            snr_estimates: Per-trace SNR in dB

        Returns:
            Denoised traces
        """
        n_samples, n_traces = traces.shape
        denoised = np.zeros_like(traces)

        step = int(self.patch_size * (1 - self.overlap))
        time_window = self.patch_size  # Use patch size as SNR window

        def process_trace_adaptive(trace_idx):
            trace = traces[:, trace_idx]
            trace_denoised = np.zeros(n_samples)
            trace_weights = np.zeros(n_samples)

            # Compute time-varying SNR using spatial statistics
            _, local_snr_windows = estimate_local_snr(
                traces, trace_idx, self.aperture, time_window
            )

            # Adaptive tolerance based on noise estimate
            trace_std = np.std(trace)
            local_tol = compute_adaptive_tolerance(
                noise_estimates[trace_idx], trace_std, self.residual_tol
            )

            for start in range(0, n_samples - self.patch_size + 1, step):
                end = start + self.patch_size
                patch = trace[start:end]

                # Get local SNR for this patch position
                window_idx = min(start // time_window, len(local_snr_windows) - 1)
                patch_snr = local_snr_windows[window_idx] if len(local_snr_windows) > 0 else 10.0

                # Compute adaptive sparsity for this patch
                patch_sparsity = compute_adaptive_sparsity(
                    patch_snr, self.sparsity,
                    self.min_sparsity, self.max_sparsity
                )

                # Normalize patch
                patch_mean = np.mean(patch)
                patch_centered = patch - patch_mean
                patch_std = np.std(patch_centered)
                if patch_std > 1e-10:
                    patch_normalized = patch_centered / patch_std
                else:
                    patch_normalized = patch_centered

                # OMP sparse coding with adaptive parameters
                Dty = self._dictionary.T @ patch_normalized
                coef, _ = omp_fast(
                    self._dictionary, patch_normalized,
                    patch_sparsity, local_tol,
                    self._gram_matrix, Dty
                )

                # Reconstruct
                reconstructed = self._dictionary @ coef

                # Denormalize
                if patch_std > 1e-10:
                    reconstructed = reconstructed * patch_std
                reconstructed = reconstructed + patch_mean

                # Accumulate with overlap weighting
                trace_denoised[start:end] += reconstructed
                trace_weights[start:end] += 1.0

            # Normalize by weights
            mask = trace_weights > 0
            trace_denoised[mask] /= trace_weights[mask]

            # Handle edges
            if not mask[0]:
                trace_denoised[:step] = trace[:step]
            if not mask[-1]:
                trace_denoised[-step:] = trace[-step:]

            return trace_denoised

        if JOBLIB_AVAILABLE and n_traces > 10:
            results = Parallel(n_jobs=N_JOBS, prefer="threads")(
                delayed(process_trace_adaptive)(i) for i in range(n_traces)
            )
            denoised = np.column_stack(results)
        else:
            for i in range(n_traces):
                denoised[:, i] = process_trace_adaptive(i)

        return denoised

    def _process_spatial(self, traces: np.ndarray) -> np.ndarray:
        """Process using spatial aperture for joint sparse coding."""
        n_samples, n_traces = traces.shape
        denoised = np.zeros_like(traces)
        weights = np.zeros_like(traces)

        step = int(self.patch_size * (1 - self.overlap))
        half_aperture = self.aperture // 2

        # Create extended dictionary for spatial processing
        # Each spatial patch is aperture * patch_size
        spatial_patch_size = self.aperture * self.patch_size
        spatial_n_atoms = self.aperture * self.n_atoms

        # Build spatial dictionary (block diagonal structure)
        spatial_dict = np.zeros((spatial_patch_size, spatial_n_atoms))
        for a in range(self.aperture):
            row_start = a * self.patch_size
            row_end = row_start + self.patch_size
            col_start = a * self.n_atoms
            col_end = col_start + self.n_atoms
            spatial_dict[row_start:row_end, col_start:col_end] = self._dictionary

        spatial_gram = spatial_dict.T @ spatial_dict
        spatial_sparsity = self.sparsity * self.aperture

        for trace_idx in range(n_traces):
            # Get aperture traces
            start_trace = max(0, trace_idx - half_aperture)
            end_trace = min(n_traces, trace_idx + half_aperture + 1)
            center_in_aperture = trace_idx - start_trace

            aperture_traces = traces[:, start_trace:end_trace]
            actual_aperture = aperture_traces.shape[1]

            for start in range(0, n_samples - self.patch_size + 1, step):
                end = start + self.patch_size

                # Extract spatial patch
                spatial_patch = aperture_traces[start:end, :].flatten('F')

                # Normalize
                patch_mean = np.mean(spatial_patch)
                patch_centered = spatial_patch - patch_mean
                patch_std = np.std(patch_centered)
                if patch_std > 1e-10:
                    patch_normalized = patch_centered / patch_std
                else:
                    patch_normalized = patch_centered

                # Adjust dictionary and do OMP
                actual_dict = spatial_dict[:actual_aperture * self.patch_size,
                                          :actual_aperture * self.n_atoms]
                actual_gram = spatial_gram[:actual_aperture * self.n_atoms,
                                          :actual_aperture * self.n_atoms]

                Dty = actual_dict.T @ patch_normalized[:actual_aperture * self.patch_size]
                coef, _ = omp_fast(
                    actual_dict,
                    patch_normalized[:actual_aperture * self.patch_size],
                    min(spatial_sparsity, actual_aperture * self.sparsity),
                    self.residual_tol,
                    actual_gram, Dty
                )

                # Reconstruct
                reconstructed = actual_dict @ coef

                # Denormalize
                if patch_std > 1e-10:
                    reconstructed = reconstructed * patch_std
                reconstructed = reconstructed + patch_mean

                # Extract center trace contribution
                center_start = center_in_aperture * self.patch_size
                center_end = center_start + self.patch_size
                center_reconstructed = reconstructed[center_start:center_end]

                denoised[start:end, trace_idx] += center_reconstructed
                weights[start:end, trace_idx] += 1.0

        # Normalize
        mask = weights > 0
        denoised[mask] /= weights[mask]

        # Handle edges
        denoised[~mask] = traces[~mask]

        return denoised
