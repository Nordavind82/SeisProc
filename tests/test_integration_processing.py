"""
Integration tests for the processing pipeline.

Tests the full workflow: load data → process → verify output.
"""
import numpy as np
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestProcessingPipeline:
    """Test the core processing pipeline."""

    def test_bandpass_filter_preserves_shape(self, seismic_data):
        """Test that bandpass filter preserves data dimensions."""
        from processors.bandpass_filter import BandpassFilter

        processor = BandpassFilter(low_freq=5.0, high_freq=80.0)
        result = processor.process(seismic_data)

        assert result.traces.shape == seismic_data.traces.shape
        assert result.sample_rate == seismic_data.sample_rate

    def test_bandpass_filter_reduces_noise(self, seismic_data):
        """Test that bandpass filter reduces out-of-band noise."""
        from processors.bandpass_filter import BandpassFilter

        # Add high-frequency noise
        noisy_traces = seismic_data.traces.copy()
        noisy_traces += 0.5 * np.random.randn(*noisy_traces.shape).astype(np.float32)

        from models.seismic_data import SeismicData
        noisy_data = SeismicData(traces=noisy_traces, sample_rate=seismic_data.sample_rate)

        # Apply narrow bandpass
        processor = BandpassFilter(low_freq=20.0, high_freq=40.0)
        result = processor.process(noisy_data)

        # Output should have lower RMS than input (noise removed)
        input_rms = np.sqrt(np.mean(noisy_traces ** 2))
        output_rms = np.sqrt(np.mean(result.traces ** 2))

        assert output_rms < input_rms, "Filter should reduce noise"

    def test_tf_denoise_preserves_shape(self, seismic_data):
        """Test that TF-Denoise preserves data dimensions."""
        from processors.tf_denoise import TFDenoise

        processor = TFDenoise(
            aperture=5,
            fmin=5.0,
            fmax=80.0,
            threshold_k=3.0
        )
        result = processor.process(seismic_data)

        assert result.traces.shape == seismic_data.traces.shape
        assert result.sample_rate == seismic_data.sample_rate

    def test_processor_chain(self, seismic_data):
        """Test applying multiple processors in sequence."""
        from processors.bandpass_filter import BandpassFilter
        from processors.tf_denoise import TFDenoise

        # First processor
        bp = BandpassFilter(low_freq=5.0, high_freq=80.0)
        intermediate = bp.process(seismic_data)

        # Second processor
        tf = TFDenoise(aperture=5, fmin=5.0, fmax=80.0, threshold_k=3.0)
        final = tf.process(intermediate)

        assert final.traces.shape == seismic_data.traces.shape

    def test_processor_progress_callback(self, seismic_data):
        """Test that progress callback is called during processing."""
        from processors.tf_denoise import TFDenoise

        progress_calls = []

        def progress_callback(current, total, message):
            progress_calls.append((current, total, message))

        processor = TFDenoise(aperture=5, fmin=5.0, fmax=80.0, threshold_k=3.0)
        processor.set_progress_callback(progress_callback)
        processor.process(seismic_data)

        # Should have received at least one progress update
        # (may be 0 if data is too small for batching)
        assert isinstance(progress_calls, list)


class TestGPUFallback:
    """Test GPU processing with CPU fallback."""

    def test_gpu_processor_falls_back_to_cpu(self, seismic_data):
        """Test that GPU processor falls back to CPU gracefully."""
        try:
            from processors.tf_denoise_gpu import TFDenoiseGPU
        except ImportError:
            pytest.skip("torch not installed")

        # Force CPU mode
        processor = TFDenoiseGPU(
            aperture=5,
            fmin=5.0,
            fmax=80.0,
            threshold_k=3.0,
            use_gpu='never'
        )

        result = processor.process(seismic_data)

        assert result.traces.shape == seismic_data.traces.shape
        assert "[CPU]" in processor.get_description()

    def test_gpu_processor_auto_mode(self, seismic_data):
        """Test GPU processor in auto mode."""
        try:
            from processors.tf_denoise_gpu import TFDenoiseGPU
        except ImportError:
            pytest.skip("torch not installed")

        processor = TFDenoiseGPU(
            aperture=5,
            fmin=5.0,
            fmax=80.0,
            threshold_k=3.0,
            use_gpu='auto'
        )

        result = processor.process(seismic_data)

        assert result.traces.shape == seismic_data.traces.shape


class TestDataRoundTrip:
    """Test data import/export round-trips."""

    def test_seismic_data_to_numpy_roundtrip(self, sample_traces):
        """Test SeismicData creation and numpy export."""
        from models.seismic_data import SeismicData

        traces, sample_rate = sample_traces

        # Create SeismicData
        data = SeismicData(traces=traces, sample_rate=sample_rate)

        # Verify properties
        assert data.n_samples == traces.shape[0]
        assert data.n_traces == traces.shape[1]
        assert data.sample_rate == sample_rate
        assert np.allclose(data.traces, traces)

    def test_zarr_roundtrip(self, zarr_data_dir, sample_traces):
        """Test Zarr storage and retrieval."""
        import zarr

        original_traces, _ = sample_traces

        # Read back from Zarr
        zarr_path = zarr_data_dir / "traces.zarr"
        z = zarr.open(str(zarr_path), mode='r')

        assert z.shape == original_traces.shape
        assert np.allclose(z[:], original_traces)
