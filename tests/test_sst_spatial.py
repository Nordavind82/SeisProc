"""
Tests for SST Denoise with spatial aperture.
"""
import pytest
import numpy as np
from models.seismic_data import SeismicData

# Check if ssqueezepy is available
try:
    from processors.sst_denoise import SSTDenoise, SSQUEEZEPY_AVAILABLE
except ImportError:
    SSQUEEZEPY_AVAILABLE = False


@pytest.fixture
def sample_seismic_data():
    """Create sample seismic data with noise."""
    np.random.seed(42)
    n_samples, n_traces = 256, 20

    # Create signal with coherent events
    traces = np.zeros((n_samples, n_traces))
    for i in range(n_traces):
        # Horizontal reflector (coherent across traces)
        for t in [50, 100, 150, 200]:
            traces[t:t+10, i] = np.sin(np.linspace(0, 2*np.pi, 10)) * 0.5

    # Add random noise
    noise = np.random.randn(n_samples, n_traces) * 0.2
    traces = traces + noise

    return SeismicData(
        traces=traces.astype(np.float32),
        sample_rate=2.0  # 2ms = 500 Hz
    )


@pytest.mark.skipif(not SSQUEEZEPY_AVAILABLE, reason="ssqueezepy not installed")
class TestSSTDenoiseValidation:
    """Test SST denoise parameter validation."""

    def test_default_params(self):
        """Test default parameters."""
        processor = SSTDenoise()
        assert processor.base_transform == 'cwt'
        assert processor.aperture == 1
        assert processor.threshold_k == 3.0
        assert processor.threshold_mode == 'soft'

    def test_invalid_aperture(self):
        """Test invalid aperture values."""
        # aperture=2 is invalid (even but >1)
        with pytest.raises(ValueError, match="Aperture must be 1"):
            SSTDenoise(aperture=2)

    def test_valid_odd_aperture(self):
        """Test valid odd aperture."""
        processor = SSTDenoise(aperture=5)
        assert processor.aperture == 5

    def test_all_threshold_modes(self):
        """Test all threshold modes are accepted."""
        for mode in ['soft', 'hard', 'scaled', 'adaptive']:
            processor = SSTDenoise(threshold_mode=mode)
            assert processor.threshold_mode == mode

    def test_invalid_threshold_mode(self):
        """Test invalid threshold mode."""
        with pytest.raises(ValueError, match="threshold_mode"):
            SSTDenoise(threshold_mode='invalid')


@pytest.mark.skipif(not SSQUEEZEPY_AVAILABLE, reason="ssqueezepy not installed")
class TestSSTDenoiseSingleTrace:
    """Test SST denoise in single-trace mode (aperture=1)."""

    def test_process_single_trace(self, sample_seismic_data):
        """Test processing with aperture=1 (single trace mode)."""
        processor = SSTDenoise(
            aperture=1,
            threshold_k=3.0,
            threshold_mode='soft'
        )

        result = processor.process(sample_seismic_data)

        assert result.traces.shape == sample_seismic_data.traces.shape
        assert result.sample_rate == sample_seismic_data.sample_rate
        # Should denoise but not zero out everything
        assert np.std(result.traces) > 0

    def test_process_stft_base(self, sample_seismic_data):
        """Test with STFT base transform (faster)."""
        processor = SSTDenoise(
            base_transform='stft',
            aperture=1,
            threshold_k=3.0
        )

        result = processor.process(sample_seismic_data)

        assert result.traces.shape == sample_seismic_data.traces.shape


@pytest.mark.skipif(not SSQUEEZEPY_AVAILABLE, reason="ssqueezepy not installed")
class TestSSTDenoiseSpatial:
    """Test SST denoise with spatial aperture."""

    def test_process_spatial_aperture(self, sample_seismic_data):
        """Test processing with spatial aperture."""
        processor = SSTDenoise(
            aperture=5,
            threshold_k=3.0,
            threshold_mode='adaptive'
        )

        result = processor.process(sample_seismic_data)

        assert result.traces.shape == sample_seismic_data.traces.shape
        assert result.sample_rate == sample_seismic_data.sample_rate

    def test_spatial_vs_single_trace(self, sample_seismic_data):
        """Test that spatial mode produces different results than single trace."""
        single_proc = SSTDenoise(aperture=1, threshold_k=3.0)
        spatial_proc = SSTDenoise(aperture=5, threshold_k=3.0)

        single_result = single_proc.process(sample_seismic_data)
        spatial_result = spatial_proc.process(sample_seismic_data)

        # Results should be different (spatial uses cross-trace statistics)
        assert not np.allclose(single_result.traces, spatial_result.traces, rtol=0.01)

    def test_aperture_larger_than_traces(self, sample_seismic_data):
        """Test when aperture is larger than number of traces."""
        n_traces = sample_seismic_data.n_traces

        processor = SSTDenoise(aperture=n_traces * 2 + 1, threshold_k=3.0)

        # Should clamp aperture and still work
        result = processor.process(sample_seismic_data)

        assert result.traces.shape == sample_seismic_data.traces.shape

    def test_time_smoothing(self, sample_seismic_data):
        """Test time smoothing parameter."""
        processor = SSTDenoise(
            aperture=5,
            time_smoothing=3,
            threshold_k=3.0
        )

        result = processor.process(sample_seismic_data)

        assert result.traces.shape == sample_seismic_data.traces.shape

    def test_low_amp_protection(self, sample_seismic_data):
        """Test low amplitude protection."""
        proc_with = SSTDenoise(
            aperture=5,
            low_amp_protection=True,
            low_amp_factor=0.3
        )
        proc_without = SSTDenoise(
            aperture=5,
            low_amp_protection=False
        )

        result_with = proc_with.process(sample_seismic_data)
        result_without = proc_without.process(sample_seismic_data)

        # Results should differ due to low-amp protection
        assert not np.allclose(result_with.traces, result_without.traces)


@pytest.mark.skipif(not SSQUEEZEPY_AVAILABLE, reason="ssqueezepy not installed")
class TestSSTDenoiseThresholdModes:
    """Test different threshold modes."""

    @pytest.mark.parametrize("mode", ['soft', 'hard', 'scaled', 'adaptive'])
    def test_threshold_mode(self, sample_seismic_data, mode):
        """Test each threshold mode produces valid output."""
        processor = SSTDenoise(
            aperture=5,
            threshold_mode=mode,
            threshold_k=3.0
        )

        result = processor.process(sample_seismic_data)

        assert result.traces.shape == sample_seismic_data.traces.shape
        assert np.isfinite(result.traces).all()


@pytest.mark.skipif(not SSQUEEZEPY_AVAILABLE, reason="ssqueezepy not installed")
class TestSSTDenoiseDescription:
    """Test description generation."""

    def test_description_single_trace(self):
        """Test description for single-trace mode."""
        processor = SSTDenoise(aperture=1, threshold_k=3.0)
        desc = processor.get_description()

        assert 'SST' in desc
        assert 'aperture' not in desc  # Single trace mode doesn't show aperture

    def test_description_spatial(self):
        """Test description for spatial mode."""
        processor = SSTDenoise(aperture=7, threshold_k=3.5, threshold_mode='adaptive')
        desc = processor.get_description()

        assert 'aperture=7' in desc
        assert 'k=3.5' in desc
        assert 'adaptive' in desc
