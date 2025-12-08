"""
Fast SEGY reader using segfast library for optimized trace loading.

Falls back to standard SEGYReader if segfast is not available.
Provides 2-10x speedup for large files.
"""
import numpy as np
from typing import Optional, List, Dict, Tuple, Callable
from pathlib import Path
import time

# Check for segfast availability
try:
    from segfast import MemmapLoader
    SEGFAST_AVAILABLE = True
except ImportError:
    SEGFAST_AVAILABLE = False
    MemmapLoader = None

from utils.segy_import.segy_reader import (
    SEGYReader,
    CancellationToken,
    OperationCancelledError,
    LoadingProgress
)
from utils.segy_import.header_mapping import HeaderMapping


class FastSEGYReader:
    """
    High-performance SEGY reader using segfast library.

    Falls back to standard SEGYReader if segfast is not installed.
    Provides significant speedup (2-10x) for large files, especially
    for random access patterns and depth slices.

    Usage:
        reader = FastSEGYReader(filename, header_mapping)
        traces, headers = reader.read_all_traces()

    Install segfast for best performance:
        pip install segfast
    """

    def __init__(self, filename: str, header_mapping: HeaderMapping):
        """
        Initialize reader with optional segfast backend.

        Args:
            filename: Path to SEGY file
            header_mapping: Header mapping configuration
        """
        self.filename = Path(filename)
        self.header_mapping = header_mapping
        self._use_segfast = SEGFAST_AVAILABLE

        if not self.filename.exists():
            raise FileNotFoundError(f"SEG-Y file not found: {filename}")

        if self._use_segfast:
            try:
                self._loader = MemmapLoader(str(self.filename))
                self._n_traces = self._loader.n_traces
                self._n_samples = self._loader.n_samples
            except Exception as e:
                # Fall back to segyio if segfast fails
                print(f"Warning: segfast initialization failed ({e}), using segyio")
                self._use_segfast = False
                self._fallback = SEGYReader(str(filename), header_mapping)
        else:
            self._fallback = SEGYReader(str(filename), header_mapping)

    @property
    def using_segfast(self) -> bool:
        """Check if segfast backend is being used."""
        return self._use_segfast

    @property
    def backend_name(self) -> str:
        """Get name of active backend."""
        return "segfast" if self._use_segfast else "segyio"

    def read_file_info(self) -> Dict[str, any]:
        """Read basic file information."""
        if self._use_segfast:
            # Get info from segfast loader
            import segyio
            with segyio.open(str(self.filename), ignore_geometry=True) as f:
                return {
                    'filename': self.filename.name,
                    'n_traces': self._n_traces,
                    'n_samples': self._n_samples,
                    'sample_interval': segyio.tools.dt(f) / 1000.0,
                    'trace_length_ms': (self._n_samples - 1) * segyio.tools.dt(f) / 1000.0,
                    'format': 'segfast',
                    'backend': 'segfast'
                }
        return self._fallback.read_file_info()

    def read_all_traces(
        self,
        max_traces: Optional[int] = None,
        cancellation_token: Optional[CancellationToken] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Read all traces efficiently using segfast.

        Args:
            max_traces: Maximum number of traces to read (None = all)
            cancellation_token: Optional token to cancel the operation
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            Tuple of (traces array, headers list)
        """
        if not self._use_segfast:
            return self._fallback.read_all_traces(
                max_traces, cancellation_token, progress_callback
            )

        n_traces = self._n_traces
        if max_traces:
            n_traces = min(n_traces, max_traces)

        # Check for cancellation
        if cancellation_token and cancellation_token.is_cancelled:
            raise OperationCancelledError("Operation cancelled")

        print(f"  Loading {n_traces:,} traces using segfast...")
        start_time = time.time()

        # Load traces in one efficient batch operation
        indices = np.arange(n_traces)
        traces = self._loader.load_traces(indices)

        # Transpose to (n_samples, n_traces) format if needed
        if traces.shape[0] == n_traces and traces.shape[1] != n_traces:
            traces = traces.T

        load_time = time.time() - start_time
        print(f"  Loaded traces in {load_time:.2f}s "
              f"({n_traces / load_time:.0f} traces/s)")

        # Load headers
        print(f"  Loading headers...")
        headers = self._load_headers_fast(indices)

        if progress_callback:
            progress_callback(n_traces, n_traces)

        return traces.astype(np.float32), headers

    def read_traces_in_chunks(
        self,
        chunk_size: int = 10000,
        cancellation_token: Optional[CancellationToken] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ):
        """
        Stream traces in chunks using segfast.

        Args:
            chunk_size: Number of traces per chunk
            cancellation_token: Optional cancellation token
            progress_callback: Optional progress callback

        Yields:
            Tuple of (traces, headers, start_idx, end_idx)
        """
        if not self._use_segfast:
            yield from self._fallback.read_traces_in_chunks(
                chunk_size, cancellation_token, progress_callback
            )
            return

        n_traces = self._n_traces
        progress = LoadingProgress(n_traces)

        for start in range(0, n_traces, chunk_size):
            if cancellation_token and cancellation_token.is_cancelled:
                raise OperationCancelledError("Cancelled")

            end = min(start + chunk_size, n_traces)
            indices = np.arange(start, end)

            # Load traces chunk
            traces = self._loader.load_traces(indices)
            if traces.shape[0] == len(indices) and len(traces.shape) > 1:
                traces = traces.T

            # Load headers
            headers = self._load_headers_fast(indices)

            # Progress update
            progress.update(end)
            if end % 10000 == 0 or end == n_traces:
                eta = progress.eta_seconds
                eta_str = f"{eta:.1f}s" if eta < float('inf') else "..."
                print(f"  Streamed {end:,}/{n_traces:,} traces "
                      f"({progress.throughput:.0f}/s, ETA: {eta_str})")
                if progress_callback:
                    progress_callback(end, n_traces)

            yield traces.astype(np.float32), headers, start, end

    def _load_headers_fast(self, indices: np.ndarray) -> List[Dict]:
        """
        Load headers for given trace indices using segfast.

        Converts segfast header format to expected dictionary format.
        """
        try:
            # Get header byte locations from mapping
            mapping = self.header_mapping.get_all_mappings()

            # segfast expects header specs as first arg, indices as second
            # Use byte locations directly
            header_specs = list(mapping.values())

            # Load headers for specified indices
            headers_df = self._loader.load_headers(
                headers=header_specs,
                indices=indices.tolist() if hasattr(indices, 'tolist') else list(indices),
                pbar=False
            )

            # Build reverse mapping: byte_loc -> our name
            byte_to_name = {v: k for k, v in mapping.items()}

            # Convert DataFrame to list of dicts
            headers = []
            for idx in range(len(headers_df)):
                header = {}
                for col in headers_df.columns:
                    # segfast columns are byte locations as ints
                    try:
                        byte_loc = int(col) if isinstance(col, (int, str)) else col
                        if byte_loc in byte_to_name:
                            name = byte_to_name[byte_loc]
                            val = headers_df.iloc[idx][col]
                            if hasattr(val, 'item'):
                                header[name] = int(val.item())
                            else:
                                header[name] = int(val)
                    except (ValueError, TypeError):
                        pass

                # Fill any missing fields
                for name in mapping.keys():
                    if name not in header:
                        header[name] = 0

                headers.append(header)

            return headers

        except Exception as e:
            # Fallback: use segyio batch header reading
            print(f"Warning: segfast header loading failed ({e}), using segyio fallback")
            return self._load_headers_segyio_fallback(indices)

    def _byte_to_column_name(self, byte_loc: int) -> str:
        """
        Convert byte location to segfast column name.

        Segfast uses standard segyio field names as column names.
        """
        import segyio

        # Map common byte locations to field names
        byte_to_field = {
            1: 'TRACE_SEQUENCE_LINE',
            5: 'TRACE_SEQUENCE_FILE',
            9: 'FieldRecord',
            13: 'TraceNumber',
            17: 'EnergySourcePoint',
            21: 'CDP',
            25: 'CDP_TRACE',
            29: 'TraceIdentificationCode',
            37: 'offset',
            41: 'ReceiverGroupElevation',
            45: 'SourceSurfaceElevation',
            69: 'SourceDepth',
            73: 'SourceX',
            77: 'SourceY',
            81: 'GroupX',
            85: 'GroupY',
            181: 'CDP_X',
            185: 'CDP_Y',
            189: 'INLINE_3D',
            193: 'CROSSLINE_3D',
        }

        if byte_loc in byte_to_field:
            return byte_to_field[byte_loc]

        # Try to find matching segyio TraceField
        try:
            for field in segyio.TraceField.enums():
                if int(field) == byte_loc:
                    return field.name
        except Exception:
            pass

        return f"byte_{byte_loc}"

    def _load_headers_segyio_fallback(self, indices: np.ndarray) -> List[Dict]:
        """
        Fallback header loading using segyio batch reading.
        Used when segfast header loading fails.
        """
        import segyio as sio

        mappings = self.header_mapping.get_all_mappings()
        n_traces = len(indices)
        start = int(indices[0]) if len(indices) > 0 else 0
        end = int(indices[-1]) + 1 if len(indices) > 0 else 0

        headers = [{} for _ in range(n_traces)]

        with sio.open(str(self.filename), 'r', ignore_geometry=True) as f:
            f.mmap()
            for name, byte_loc in mappings.items():
                try:
                    values = f.attributes(byte_loc)[start:end]
                    for i, val in enumerate(values):
                        headers[i][name] = int(val)
                except (KeyError, IndexError, TypeError):
                    for i in range(n_traces):
                        headers[i][name] = 0

        return headers

    def detect_ensemble_boundaries(self, headers: List[Dict]) -> List[Tuple[int, int]]:
        """Detect ensemble boundaries (delegates to standard reader)."""
        if self._use_segfast:
            # Use SEGYReader for ensemble detection
            temp_reader = SEGYReader(str(self.filename), self.header_mapping)
            return temp_reader.detect_ensemble_boundaries(headers)
        return self._fallback.detect_ensemble_boundaries(headers)

    def __repr__(self) -> str:
        backend = "segfast" if self._use_segfast else "segyio"
        return f"FastSEGYReader('{self.filename.name}', backend={backend})"


def create_segy_reader(
    filename: str,
    header_mapping: HeaderMapping,
    prefer_fast: bool = True
) -> 'SEGYReader':
    """
    Factory function to create appropriate SEGY reader.

    Args:
        filename: Path to SEGY file
        header_mapping: Header mapping configuration
        prefer_fast: If True, use FastSEGYReader when segfast available

    Returns:
        SEGY reader instance (FastSEGYReader or SEGYReader)
    """
    if prefer_fast and SEGFAST_AVAILABLE:
        try:
            return FastSEGYReader(filename, header_mapping)
        except Exception as e:
            print(f"Warning: FastSEGYReader init failed ({e}), using standard reader")

    return SEGYReader(filename, header_mapping)


# Convenience function to check if fast reader is available
def is_fast_reader_available() -> bool:
    """Check if segfast is available for fast reading."""
    return SEGFAST_AVAILABLE
