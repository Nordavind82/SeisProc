"""Quick debug script to identify the shape mismatch issue."""
import numpy as np
import zarr
import json
import pandas as pd
from pathlib import Path
import tempfile
from numcodecs import Blosc
from models.lazy_seismic_data import LazySeismicData


def create_minimal_test():
    """Create minimal test case to debug shape issue."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create minimal Zarr array
        n_samples = 500
        n_traces = 1000
        sample_rate = 2.0

        zarr_path = output_dir / 'traces.zarr'
        print(f"Creating Zarr array: shape=({n_samples}, {n_traces})")

        z = zarr.open_array(
            str(zarr_path),
            mode='w',
            shape=(n_samples, n_traces),
            chunks=(n_samples, min(1000, n_traces)),
            dtype=np.float32,
            compressor=Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE),
            zarr_format=2
        )

        # Fill with known pattern
        for i in range(n_traces):
            z[:, i] = float(i)

        print(f"Zarr array created: shape={z.shape}")

        # Create metadata
        metadata = {
            'shape': [n_samples, n_traces],
            'sample_rate': sample_rate,
            'n_samples': n_samples,
            'n_traces': n_traces,
            'duration_ms': (n_samples - 1) * sample_rate,
            'nyquist_freq': 1000.0 / (2.0 * sample_rate),
        }

        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f)

        # Create ensemble index
        ensembles = []
        for i in range(n_traces // 100):
            ensembles.append({
                'ensemble_id': i,
                'start_trace': i * 100,
                'end_trace': (i + 1) * 100 - 1,
                'n_traces': 100,
                'cdp': i + 1,
            })

        df_ensembles = pd.DataFrame(ensembles)
        df_ensembles.to_parquet(output_dir / 'ensemble_index.parquet', compression='snappy', index=False)

        print("\nLoading as LazySeismicData...")
        lazy_data = LazySeismicData.from_storage_dir(str(output_dir))

        print(f"  n_samples: {lazy_data.n_samples}")
        print(f"  n_traces: {lazy_data.n_traces}")
        print(f"  sample_rate: {lazy_data.sample_rate}")
        print(f"  duration: {lazy_data.duration}")

        # Test get_trace_range
        print(f"\nTesting get_trace_range(100, 150)...")
        traces = lazy_data.get_trace_range(100, 150)
        print(f"  Returned shape: {traces.shape}")
        print(f"  Expected shape: ({n_samples}, 50)")

        if traces.shape[0] != n_samples:
            print(f"  ERROR: Got {traces.shape[0]} samples instead of {n_samples}")

            # Debug the get_window call
            print(f"\n  Debugging get_window call...")
            time_arg = n_samples * sample_rate
            print(f"    Time argument: {time_arg}")
            print(f"    sample_end = int({time_arg} / {sample_rate}) = {int(time_arg / sample_rate)}")

            # Try direct window call
            window = lazy_data.get_window(0, time_arg, 100, 150)
            print(f"    Direct get_window(0, {time_arg}, 100, 150) shape: {window.shape}")

        # Test get_ensemble
        print(f"\nTesting get_ensemble(5)...")
        ensemble = lazy_data.get_ensemble(5)
        print(f"  Returned shape: {ensemble.shape}")
        print(f"  Expected shape: ({n_samples}, 100)")

        if ensemble.shape[0] != n_samples:
            print(f"  ERROR: Got {ensemble.shape[0]} samples instead of {n_samples}")


if __name__ == '__main__':
    create_minimal_test()
