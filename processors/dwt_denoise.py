"""
DWT-Denoise processor using Discrete Wavelet Transform with MAD-based thresholding.

Implements wavelet shrinkage denoising for seismic data with:
- Multiple wavelet families (Daubechies, Symlet, Coiflet, etc.)
- Stationary Wavelet Transform (SWT) for translation-invariant denoising
- Wavelet Packet Transform (WPT) for adaptive frequency band selection
- Spatial aperture processing for robust noise estimation
- MAD-based adaptive thresholding

Key advantages over STFT:
- 5-10x faster processing
- Better SNR improvement for random noise
- Multi-resolution analysis naturally suited for seismic signals
- Perfect reconstruction guarantee

WPT advantages over DWT:
- Full binary tree decomposition (both approximation and detail)
- Better frequency resolution
- Best-basis selection for optimal denoising
- Ideal for narrowband noise (ground roll, powerline harmonics)
"""
import numpy as np
from typing import Optional, Literal
import logging
from models.seismic_data import SeismicData
from processors.base_processor import BaseProcessor

logger = logging.getLogger(__name__)

# Try to import pywt
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    logger.warning("PyWavelets not available. Install with: pip install PyWavelets")


class DWTDenoise(BaseProcessor):
    """
    Discrete Wavelet Transform Denoising using wavelet shrinkage.

    Implements wavelet-based denoising with MAD thresholding for
    effective random noise attenuation while preserving signal.

    Supports multiple modes:
    - 'dwt': Standard DWT (fast, but has shift-variance artifacts)
    - 'swt': Stationary WT (slower, but translation-invariant)
    - 'dwt_spatial': DWT with spatial aperture for robust thresholding
    - 'wpt': Wavelet Packet Transform (full tree, better frequency resolution)
    - 'wpt_spatial': WPT with spatial aperture processing

    Typical speedup: 5-10x faster than STFT with comparable or better SNR.
    """

    def __init__(self,
                 wavelet: str = 'db4',
                 level: Optional[int] = None,
                 threshold_k: float = 3.0,
                 threshold_mode: Literal['soft', 'hard'] = 'soft',
                 transform_type: Literal['dwt', 'swt', 'dwt_spatial', 'wpt', 'wpt_spatial'] = 'dwt',
                 aperture: int = 7,
                 best_basis: bool = False):
        """
        Initialize DWT-Denoise processor.

        Args:
            wavelet: Wavelet to use. Common choices:
                - 'db4', 'db8': Daubechies wavelets (good general purpose)
                - 'sym4', 'sym8': Symlets (more symmetric)
                - 'coif4': Coiflets (nearly symmetric)
                - 'bior3.5': Biorthogonal (linear phase)
            level: Decomposition level (None = automatic based on signal length)
            threshold_k: MAD threshold multiplier (default 3.0)
            threshold_mode: 'soft' (wavelet shrinkage) or 'hard' (keep/zero)
            transform_type: Transform variant:
                - 'dwt': Fast standard DWT (default)
                - 'swt': Stationary WT (translation-invariant, slower)
                - 'dwt_spatial': DWT with spatial aperture processing
                - 'wpt': Wavelet Packet Transform (full tree decomposition)
                - 'wpt_spatial': WPT with spatial aperture processing
            aperture: Spatial aperture size for spatial modes (odd number)
            best_basis: For WPT modes, use best-basis selection (Shannon entropy)
        """
        if not PYWT_AVAILABLE:
            raise ImportError("PyWavelets required. Install with: pip install PyWavelets")

        self.wavelet = wavelet
        self.level = level
        self.threshold_k = threshold_k
        self.threshold_mode = threshold_mode
        self.transform_type = transform_type
        self.aperture = aperture
        self.best_basis = best_basis

        super().__init__(
            wavelet=wavelet,
            level=level,
            threshold_k=threshold_k,
            threshold_mode=threshold_mode,
            transform_type=transform_type,
            aperture=aperture,
            best_basis=best_basis
        )

    def _validate_params(self):
        """Validate processor parameters."""
        if self.wavelet not in pywt.wavelist():
            raise ValueError(f"Unknown wavelet '{self.wavelet}'. Use pywt.wavelist() for options.")
        if self.threshold_k <= 0:
            raise ValueError("threshold_k must be positive")
        if self.threshold_mode not in ['soft', 'hard']:
            raise ValueError("threshold_mode must be 'soft' or 'hard'")
        if self.transform_type not in ['dwt', 'swt', 'dwt_spatial', 'wpt', 'wpt_spatial']:
            raise ValueError("transform_type must be 'dwt', 'swt', 'dwt_spatial', 'wpt', or 'wpt_spatial'")
        if self.transform_type in ['dwt_spatial', 'wpt_spatial']:
            if self.aperture < 3:
                raise ValueError("Aperture must be at least 3")
            if self.aperture % 2 == 0:
                raise ValueError("Aperture must be odd")

    def get_description(self) -> str:
        """Get processor description."""
        desc = (f"DWT-Denoise ({self.transform_type.upper()}): "
                f"wavelet={self.wavelet}, "
                f"k={self.threshold_k:.1f}, "
                f"{self.threshold_mode} threshold")
        if self.transform_type in ['wpt', 'wpt_spatial'] and self.best_basis:
            desc += ", best-basis"
        return desc

    def process(self, data: SeismicData) -> SeismicData:
        """
        Apply DWT-domain denoising to seismic data.

        Args:
            data: Input seismic data

        Returns:
            Denoised seismic data
        """
        import time

        start_time = time.time()

        # Use direct reference - processing methods don't modify input
        # This saves ~20MB per gather (one fewer copy in memory)
        traces = data.traces
        n_samples, n_traces = traces.shape

        # Determine decomposition level
        if self.level is None:
            max_level = pywt.dwt_max_level(n_samples, self.wavelet)
            self.level = min(max_level, 6)  # Cap at 6 for efficiency

        logger.info(
            f"DWT-Denoise: {n_traces} traces × {n_samples} samples | "
            f"Wavelet: {self.wavelet} | Level: {self.level} | "
            f"Type: {self.transform_type} | k={self.threshold_k}"
        )

        # Process based on transform type
        if self.transform_type == 'dwt':
            denoised_traces = self._process_dwt(traces)
        elif self.transform_type == 'swt':
            denoised_traces = self._process_swt(traces)
        elif self.transform_type == 'wpt':
            denoised_traces = self._process_wpt(traces)
        elif self.transform_type == 'wpt_spatial':
            denoised_traces = self._process_wpt_spatial(traces)
        else:  # dwt_spatial
            denoised_traces = self._process_dwt_spatial(traces)

        # Compute metrics
        elapsed = time.time() - start_time
        throughput = n_traces / elapsed if elapsed > 0 else 0

        input_rms = np.sqrt(np.mean(traces**2))
        output_rms = np.sqrt(np.mean(denoised_traces**2))
        energy_ratio = output_rms / input_rms if input_rms > 0 else 0

        logger.info(
            f"DWT-Denoise complete: {elapsed*1000:.1f}ms | "
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

    def _process_dwt(self, traces: np.ndarray) -> np.ndarray:
        """
        Fast DWT denoising - process each trace independently.

        Uses MAD estimation from finest detail coefficients.
        """
        n_samples, n_traces = traces.shape
        denoised = np.zeros_like(traces)

        for i in range(n_traces):
            coeffs = pywt.wavedec(traces[:, i], self.wavelet, level=self.level)

            # Estimate noise from finest level detail coefficients
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = self.threshold_k * sigma

            # Threshold detail coefficients (keep approximation unchanged)
            denoised_coeffs = [coeffs[0]]
            for c in coeffs[1:]:
                denoised_coeffs.append(
                    pywt.threshold(c, threshold, mode=self.threshold_mode)
                )

            # Reconstruct
            denoised[:, i] = pywt.waverec(denoised_coeffs, self.wavelet)[:n_samples]

        return denoised

    def _process_swt(self, traces: np.ndarray) -> np.ndarray:
        """
        Stationary Wavelet Transform denoising - translation invariant.

        Slower but avoids shift-variance artifacts of standard DWT.
        """
        n_samples, n_traces = traces.shape

        # Pad to power of 2 for SWT
        target_len = 2**int(np.ceil(np.log2(n_samples)))
        pad_len = target_len - n_samples

        if pad_len > 0:
            traces_padded = np.pad(traces, ((0, pad_len), (0, 0)), mode='reflect')
        else:
            traces_padded = traces

        denoised_padded = np.zeros_like(traces_padded)

        # Adjust level for SWT
        max_level = pywt.swt_max_level(traces_padded.shape[0])
        level = min(self.level, max_level)

        for i in range(n_traces):
            coeffs = pywt.swt(traces_padded[:, i], self.wavelet, level=level)

            # Estimate noise from finest level detail
            sigma = np.median(np.abs(coeffs[0][1])) / 0.6745
            threshold = self.threshold_k * sigma

            # Threshold detail coefficients
            denoised_coeffs = []
            for cA, cD in coeffs:
                cD_thresh = pywt.threshold(cD, threshold, mode=self.threshold_mode)
                denoised_coeffs.append((cA, cD_thresh))

            denoised_padded[:, i] = pywt.iswt(denoised_coeffs, self.wavelet)

        return denoised_padded[:n_samples, :]

    def _process_dwt_spatial(self, traces: np.ndarray) -> np.ndarray:
        """
        DWT denoising with spatial aperture processing.

        Uses spatial MAD estimation across neighboring traces for
        more robust threshold calculation.
        """
        n_samples, n_traces = traces.shape
        denoised = np.zeros_like(traces)
        half_ap = self.aperture // 2

        for trace_idx in range(n_traces):
            # Get spatial aperture
            start_idx = max(0, trace_idx - half_ap)
            end_idx = min(n_traces, trace_idx + half_ap + 1)
            ensemble = traces[:, start_idx:end_idx]
            center_in_ensemble = trace_idx - start_idx

            # DWT decomposition for all traces in aperture
            all_coeffs = []
            for i in range(ensemble.shape[1]):
                coeffs = pywt.wavedec(ensemble[:, i], self.wavelet, level=self.level)
                all_coeffs.append(coeffs)

            # Get center trace coefficients
            center_coeffs = all_coeffs[center_in_ensemble]

            # Compute MAD-based threshold for each level using spatial statistics
            denoised_coeffs = [center_coeffs[0]]  # Keep approximation

            for level_idx in range(1, len(center_coeffs)):
                # Collect this level's coefficients from all traces
                level_coeffs = np.array([c[level_idx] for c in all_coeffs])

                # Compute spatial MAD
                median_coef = np.median(level_coeffs, axis=0)
                mad = np.median(np.abs(level_coeffs - median_coef), axis=0)
                mad_scaled = mad * 1.4826

                # Prevent zero threshold
                min_mad = np.maximum(0.01 * np.abs(median_coef), 1e-10)
                mad_scaled = np.maximum(mad_scaled, min_mad)

                threshold = self.threshold_k * mad_scaled

                # Apply thresholding
                center_coef = center_coeffs[level_idx]
                if self.threshold_mode == 'soft':
                    denoised_coef = np.sign(center_coef) * np.maximum(
                        np.abs(center_coef) - threshold, 0
                    )
                else:  # hard
                    denoised_coef = np.where(
                        np.abs(center_coef) > threshold, center_coef, 0
                    )

                denoised_coeffs.append(denoised_coef)

            # Reconstruct
            denoised[:, trace_idx] = pywt.waverec(
                denoised_coeffs, self.wavelet
            )[:n_samples]

        return denoised

    def _process_wpt(self, traces: np.ndarray) -> np.ndarray:
        """
        Wavelet Packet Transform denoising - full binary tree decomposition.

        WPT decomposes both approximation and detail coefficients at each level,
        providing better frequency resolution than standard DWT.

        Optionally uses best-basis selection (Shannon entropy) to find the
        optimal decomposition tree for the given signal.
        """
        n_samples, n_traces = traces.shape
        denoised = np.zeros_like(traces)

        for i in range(n_traces):
            # Create wavelet packet tree
            wp = pywt.WaveletPacket(
                data=traces[:, i],
                wavelet=self.wavelet,
                mode='symmetric',
                maxlevel=self.level
            )

            # Get all nodes at the deepest level (leaf nodes)
            if self.best_basis:
                # Use Shannon entropy for best-basis selection
                # Get the best basis nodes
                nodes = self._get_best_basis_nodes(wp)
            else:
                # Use all leaf nodes at maximum level
                nodes = [node.path for node in wp.get_level(self.level, 'natural')]

            # Estimate noise from all coefficients at deepest level
            all_coeffs = []
            for path in nodes:
                node = wp[path]
                if node.data is not None:
                    all_coeffs.extend(node.data.flatten())

            if len(all_coeffs) > 0:
                sigma = np.median(np.abs(all_coeffs)) / 0.6745
                threshold = self.threshold_k * sigma
            else:
                threshold = 0

            # Threshold all nodes
            for path in nodes:
                node = wp[path]
                if node.data is not None and len(node.data) > 0:
                    node.data = pywt.threshold(
                        node.data, threshold, mode=self.threshold_mode
                    )

            # Reconstruct
            denoised[:, i] = wp.reconstruct(update=True)[:n_samples]

        return denoised

    def _process_wpt_spatial(self, traces: np.ndarray) -> np.ndarray:
        """
        Wavelet Packet Transform denoising with spatial aperture processing.

        Uses spatial MAD estimation across neighboring traces for
        more robust threshold calculation at each WPT node.
        """
        n_samples, n_traces = traces.shape
        denoised = np.zeros_like(traces)
        half_ap = self.aperture // 2

        for trace_idx in range(n_traces):
            # Get spatial aperture
            start_idx = max(0, trace_idx - half_ap)
            end_idx = min(n_traces, trace_idx + half_ap + 1)
            ensemble = traces[:, start_idx:end_idx]
            center_in_ensemble = trace_idx - start_idx

            # WPT decomposition for all traces in aperture
            all_wps = []
            for j in range(ensemble.shape[1]):
                wp = pywt.WaveletPacket(
                    data=ensemble[:, j],
                    wavelet=self.wavelet,
                    mode='symmetric',
                    maxlevel=self.level
                )
                all_wps.append(wp)

            # Get center wavelet packet
            center_wp = all_wps[center_in_ensemble]

            # Get nodes to process
            if self.best_basis:
                nodes = self._get_best_basis_nodes(center_wp)
            else:
                nodes = [node.path for node in center_wp.get_level(self.level, 'natural')]

            # Process each node with spatial MAD thresholding
            for path in nodes:
                # Collect coefficients from all traces for this node
                node_coeffs = []
                for wp in all_wps:
                    node = wp[path]
                    if node.data is not None:
                        node_coeffs.append(node.data)

                if not node_coeffs:
                    continue

                # Stack into array for spatial statistics
                node_coeffs = np.array(node_coeffs)

                # Compute spatial MAD
                median_coef = np.median(node_coeffs, axis=0)
                mad = np.median(np.abs(node_coeffs - median_coef), axis=0)
                mad_scaled = mad * 1.4826

                # Prevent zero threshold
                min_mad = np.maximum(0.01 * np.abs(median_coef), 1e-10)
                mad_scaled = np.maximum(mad_scaled, min_mad)

                threshold = self.threshold_k * mad_scaled

                # Apply thresholding to center trace
                center_node = center_wp[path]
                if center_node.data is not None:
                    center_coef = center_node.data
                    if self.threshold_mode == 'soft':
                        center_node.data = np.sign(center_coef) * np.maximum(
                            np.abs(center_coef) - threshold, 0
                        )
                    else:  # hard
                        center_node.data = np.where(
                            np.abs(center_coef) > threshold, center_coef, 0
                        )

            # Reconstruct
            denoised[:, trace_idx] = center_wp.reconstruct(update=True)[:n_samples]

        return denoised

    def _get_best_basis_nodes(self, wp) -> list:
        """
        Get best-basis nodes using Shannon entropy criterion.

        The best-basis algorithm finds the optimal subtree that minimizes
        the total entropy of the decomposition coefficients.

        Args:
            wp: WaveletPacket object

        Returns:
            List of node paths for the best basis
        """
        def shannon_entropy(data):
            """Compute Shannon entropy of normalized energy distribution."""
            if data is None or len(data) == 0:
                return 0
            energy = data ** 2
            total_energy = np.sum(energy)
            if total_energy == 0:
                return 0
            p = energy / total_energy
            # Avoid log(0)
            p = p[p > 0]
            return -np.sum(p * np.log(p))

        def best_basis_recursive(node_path):
            """Recursively find best basis using bottom-up algorithm."""
            node = wp[node_path]

            if node.level >= self.level:
                # Leaf node - return its entropy and path
                return shannon_entropy(node.data), [node_path]

            # Get children
            left_path = node_path + 'a'
            right_path = node_path + 'd'

            # Recursively get best basis for children
            try:
                left_entropy, left_nodes = best_basis_recursive(left_path)
                right_entropy, right_nodes = best_basis_recursive(right_path)
                children_entropy = left_entropy + right_entropy
                children_nodes = left_nodes + right_nodes
            except (KeyError, AttributeError):
                # Children don't exist, use current node
                return shannon_entropy(node.data), [node_path]

            # Compare parent vs children
            parent_entropy = shannon_entropy(node.data)

            if parent_entropy <= children_entropy:
                # Parent is better - use it
                return parent_entropy, [node_path]
            else:
                # Children are better - use them
                return children_entropy, children_nodes

        # Start from root
        _, best_nodes = best_basis_recursive('')

        # Filter out empty nodes
        result = []
        for path in best_nodes:
            node = wp[path]
            if node.data is not None and len(node.data) > 0:
                result.append(path)

        return result if result else [node.path for node in wp.get_level(self.level, 'natural')]
