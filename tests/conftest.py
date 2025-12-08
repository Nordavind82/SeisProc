"""
Pytest configuration and fixtures for SeisProc tests.
"""
import numpy as np
import pytest
import tempfile
import shutil
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_traces():
    """Generate sample seismic trace data."""
    np.random.seed(42)
    n_samples = 1000
    n_traces = 50
    sample_rate = 2.0  # ms

    # Create synthetic data with signal and noise
    t = np.arange(n_samples) * sample_rate / 1000.0  # time in seconds

    traces = np.zeros((n_samples, n_traces), dtype=np.float32)
    for i in range(n_traces):
        # Add a Ricker wavelet at random time
        center = 0.2 + 0.01 * i  # moveout
        freq = 30  # Hz
        wavelet_t = t - center
        wavelet = (1 - 2 * (np.pi * freq * wavelet_t) ** 2) * np.exp(-(np.pi * freq * wavelet_t) ** 2)
        traces[:, i] = wavelet + 0.1 * np.random.randn(n_samples)

    return traces, sample_rate


@pytest.fixture
def seismic_data(sample_traces):
    """Create SeismicData instance from sample traces."""
    from models.seismic_data import SeismicData

    traces, sample_rate = sample_traces
    return SeismicData(traces=traces, sample_rate=sample_rate)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sample_segy_path(temp_dir, sample_traces):
    """Create a sample SEG-Y file for testing."""
    try:
        import segyio
    except ImportError:
        pytest.skip("segyio not installed")

    traces, sample_rate = sample_traces
    n_samples, n_traces = traces.shape

    segy_path = temp_dir / "test.sgy"

    # Create SEG-Y spec
    spec = segyio.spec()
    spec.format = 1  # IBM float
    spec.samples = range(n_samples)
    spec.tracecount = n_traces

    with segyio.create(str(segy_path), spec) as f:
        for i in range(n_traces):
            f.trace[i] = traces[:, i]
            f.header[i][segyio.TraceField.TRACE_SEQUENCE_LINE] = i + 1
            f.header[i][segyio.TraceField.FieldRecord] = 100
            f.header[i][segyio.TraceField.CDP] = 1000 + i

    return segy_path


@pytest.fixture
def zarr_data_dir(temp_dir, sample_traces):
    """Create a Zarr data directory for testing."""
    import zarr
    import pandas as pd

    traces, sample_rate = sample_traces
    n_samples, n_traces = traces.shape

    # Create Zarr array
    zarr_path = temp_dir / "traces.zarr"
    z = zarr.open(str(zarr_path), mode='w', shape=traces.shape,
                  chunks=(n_samples, 100), dtype='float32')
    z[:] = traces

    # Create headers parquet
    headers = pd.DataFrame({
        'TRACE_SEQUENCE_LINE': range(1, n_traces + 1),
        'FieldRecord': [100] * n_traces,
        'CDP': range(1000, 1000 + n_traces),
    })
    headers.to_parquet(temp_dir / "headers.parquet")

    # Create metadata
    import json
    metadata = {
        'sample_rate_ms': sample_rate,
        'n_samples': n_samples,
        'n_traces': n_traces,
    }
    with open(temp_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f)

    return temp_dir
