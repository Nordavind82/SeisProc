"""
Data storage manager for Zarr (traces) and Parquet (headers).
Efficient storage and fast access for large seismic datasets.

Performance optimizations for large datasets (>1M traces):
- Periodic garbage collection
- Smaller header batch sizes to reduce memory pressure
- Explicit memory cleanup after batch writes
"""
import gc
import numpy as np
import zarr
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import sys
from models.seismic_data import SeismicData

# GC threshold for large dataset processing
GC_INTERVAL = 50000  # Run GC every N traces


class DataStorage:
    """
    Manages efficient storage of seismic data and headers.

    Storage structure:
        output_dir/
            ├── traces.zarr/          # Trace data (compressed)
            ├── headers.parquet       # All trace headers
            ├── ensemble_index.parquet # Ensemble boundaries
            ├── trace_index.parquet   # Trace indices
            └── metadata.json         # File and processing metadata
    """

    def __init__(self, output_dir: str):
        """
        Initialize data storage.

        Args:
            output_dir: Directory for storing data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.traces_path = self.output_dir / 'traces.zarr'
        self.headers_path = self.output_dir / 'headers.parquet'
        self.ensemble_index_path = self.output_dir / 'ensemble_index.parquet'
        self.trace_index_path = self.output_dir / 'trace_index.parquet'
        self.metadata_path = self.output_dir / 'metadata.json'

    def save_seismic_data(
        self,
        seismic_data: SeismicData,
        headers: List[Dict],
        ensembles: List[Tuple[int, int]],
        chunk_size: int = 1000
    ):
        """
        Save seismic data, headers, and ensemble boundaries.

        Args:
            seismic_data: SeismicData object with traces
            headers: List of header dictionaries
            ensembles: List of (start, end) ensemble boundaries
            chunk_size: Chunk size for Zarr compression
        """
        print(f"Saving data to: {self.output_dir}")

        # 1. Save traces to Zarr (compressed)
        self._save_traces_zarr(seismic_data.traces, chunk_size)

        # 2. Save headers to Parquet
        self._save_headers_parquet(headers)

        # 3. Save trace index
        self._save_trace_index(seismic_data.n_traces)

        # 4. Save ensemble index (pass n_traces for default ensemble if needed)
        self._save_ensemble_index(ensembles, n_traces=seismic_data.n_traces)

        # 5. Save metadata
        self._save_metadata(seismic_data)

        print(f"✓ Data saved successfully")

    def save_traces_streaming(
        self,
        trace_generator,
        n_samples: int,
        n_traces: int,
        chunk_size: int = 5000,
        progress_callback=None
    ):
        """
        Save traces from a streaming generator directly to Zarr without loading full dataset.

        This method enables memory-efficient storage of large datasets by writing chunks
        directly to Zarr as they are generated, without accumulating them in memory.

        Args:
            trace_generator: Generator yielding (traces_chunk, headers_chunk, start_idx, end_idx)
            n_samples: Number of samples per trace (time dimension)
            n_traces: Total number of traces in dataset
            chunk_size: Zarr chunk size for storage (default: 5000)
            progress_callback: Optional callback function(current_trace, total_traces)

        Returns:
            Compression ratio achieved

        Memory Usage:
            O(chunk_size) - Only one chunk in memory at a time

        Example:
            >>> storage = DataStorage('output_dir')
            >>> reader = SEGYReader('file.sgy', mapping)
            >>> generator = reader.read_traces_in_chunks(chunk_size=5000)
            >>> storage.save_traces_streaming(generator, n_samples=1000, n_traces=50000)
        """
        print(f"  Creating Zarr array: shape=({n_samples}, {n_traces})")

        # Create Zarr array with compression upfront (using zarr v2 format)
        # No compression for maximum write speed
        # Trade-off: ~3-4x larger files but ~2-3x faster writes
        z = zarr.open(
            str(self.traces_path),
            mode='w',
            shape=(n_samples, n_traces),
            chunks=(n_samples, min(chunk_size, 1000)),  # Chunk along trace dimension
            dtype=np.float32,
            compressor=None,
            zarr_format=2
        )

        print(f"  Streaming traces to Zarr...")
        total_written = 0
        uncompressed_size = 0

        # Write chunks as they come from generator
        for traces_chunk, headers_chunk, start_idx, end_idx in trace_generator:
            # Write chunk at correct offset
            z[:, start_idx:end_idx] = traces_chunk

            # Track progress
            total_written = end_idx
            uncompressed_size += traces_chunk.nbytes

            # Progress callback
            if progress_callback:
                progress_callback(total_written, n_traces)

            # Progress feedback
            if total_written % 5000 == 0 or total_written == n_traces:
                print(f"    Written {total_written}/{n_traces} traces to Zarr...")

        # Calculate compression ratio
        compressed_size = sum(f.stat().st_size for f in self.traces_path.rglob('*') if f.is_file())
        compression_ratio = uncompressed_size / compressed_size if compressed_size > 0 else 1.0

        print(f"    Zarr compression ratio: {compression_ratio:.2f}x")
        print(f"    Uncompressed: {uncompressed_size / 1024 / 1024:.1f} MB")
        print(f"    Compressed: {compressed_size / 1024 / 1024:.1f} MB")

        return compression_ratio

    def save_headers_streaming(
        self,
        header_generator,
        batch_size: int = 10000,
        progress_callback=None
    ):
        """
        Save headers from a streaming generator to Parquet without loading all in memory.

        This method accumulates headers in batches and writes them to Parquet incrementally,
        enabling memory-efficient storage of large header datasets.

        Args:
            header_generator: Generator yielding (traces_chunk, headers_chunk, start_idx, end_idx)
            batch_size: Number of headers to accumulate before writing (default: 10000)
            progress_callback: Optional callback function(current_trace, total_traces)

        Returns:
            Total number of headers saved

        Memory Usage:
            O(batch_size) - Only one batch in memory at a time

        Example:
            >>> storage = DataStorage('output_dir')
            >>> reader = SEGYReader('file.sgy', mapping)
            >>> generator = reader.read_traces_in_chunks(chunk_size=5000)
            >>> storage.save_headers_streaming(generator, batch_size=10000)
        """
        print(f"  Streaming headers to Parquet (batch size: {batch_size})...")

        header_buffer = []
        total_headers = 0
        trace_index_offset = 0

        # Process headers from generator
        for traces_chunk, headers_chunk, start_idx, end_idx in header_generator:
            # Add headers to buffer
            header_buffer.extend(headers_chunk)

            # Write batch when buffer is full
            if len(header_buffer) >= batch_size:
                self._write_header_batch(header_buffer, trace_index_offset, append=(total_headers > 0))
                total_headers += len(header_buffer)
                trace_index_offset += len(header_buffer)
                header_buffer = []

                # Progress callback
                if progress_callback:
                    progress_callback(total_headers, None)

                print(f"    Written {total_headers} headers to Parquet...")

        # Write remaining headers
        if header_buffer:
            self._write_header_batch(header_buffer, trace_index_offset, append=(total_headers > 0))
            total_headers += len(header_buffer)
            print(f"    Written {total_headers} headers to Parquet...")

        # Merge all header chunk files
        self._finalize_headers()

        # Final progress callback
        if progress_callback:
            progress_callback(total_headers, total_headers)

        print(f"    Completed: {total_headers} headers saved")
        return total_headers

    def _write_header_batch(self, headers: List[Dict], trace_index_offset: int, append: bool = False):
        """
        Write a batch of headers to Parquet efficiently.

        Uses chunked writing to avoid O(n²) behavior:
        - First batch creates the main file
        - Subsequent batches write to chunk files
        - Chunks are merged at the end via _finalize_headers()

        Args:
            headers: List of header dictionaries
            trace_index_offset: Starting trace index for this batch
            append: If True, write to chunk file; if False, create main file
        """
        if not headers:
            return

        import pyarrow as pa
        import pyarrow.parquet as pq

        # Convert to DataFrame
        df_headers = pd.DataFrame(headers)

        # Add trace index column
        df_headers['trace_index'] = np.arange(trace_index_offset, trace_index_offset + len(headers))

        # Convert to Arrow table
        table = pa.Table.from_pandas(df_headers, preserve_index=False)

        if not append:
            # First batch - create the main file
            pq.write_table(table, self.headers_path, compression='snappy')
            # Store schema for subsequent chunks
            self._header_schema = table.schema
            self._header_chunk_files = []
        else:
            # Subsequent batches - write to chunk files (much faster than read-modify-write)
            chunk_path = self.output_dir / f"_headers_chunk_{trace_index_offset}.parquet"
            pq.write_table(table, chunk_path, compression='snappy')
            if not hasattr(self, '_header_chunk_files'):
                self._header_chunk_files = []
            self._header_chunk_files.append(chunk_path)

        # Clean up DataFrame reference
        del df_headers
        del table

    def _finalize_headers(self):
        """
        Merge all header chunk files into the main headers.parquet file.

        This is called at the end of streaming import to consolidate all chunks.
        Uses efficient concatenation rather than repeated read-modify-write.
        """
        import pyarrow.parquet as pq
        import pyarrow as pa

        if not hasattr(self, '_header_chunk_files') or not self._header_chunk_files:
            return  # No chunks to merge

        print(f"  Merging {len(self._header_chunk_files)} header chunks...")

        # Read all tables
        tables = [pq.read_table(self.headers_path)]  # Main file first
        for chunk_path in self._header_chunk_files:
            tables.append(pq.read_table(chunk_path))

        # Concatenate all tables
        merged_table = pa.concat_tables(tables)

        # Write merged file
        pq.write_table(merged_table, self.headers_path, compression='snappy')

        # Clean up chunk files
        for chunk_path in self._header_chunk_files:
            try:
                chunk_path.unlink()
            except Exception:
                pass

        self._header_chunk_files = []

        # Release memory
        del tables
        del merged_table
        gc.collect()

        print(f"  Headers merged successfully.")

    def _save_traces_zarr(self, traces: np.ndarray, chunk_size: int):
        """Save trace data to Zarr with compression."""
        print(f"  Saving traces to Zarr: {traces.shape}")

        # Create Zarr array with compression (using zarr v2 format for compatibility)
        # No compression for maximum write speed
        z = zarr.open(
            str(self.traces_path),
            mode='w',
            shape=traces.shape,
            chunks=(traces.shape[0], chunk_size),  # Chunk along trace dimension
            dtype=traces.dtype,
            compressor=None,
            zarr_format=2  # Use Zarr v2 format for compatibility
        )

        # Write data
        z[:] = traces

        print(f"    Zarr compression ratio: {traces.nbytes / self.traces_path.stat().st_size:.2f}x")

    def save_all_streaming(
        self,
        trace_generator,
        n_samples: int,
        n_traces: int,
        ensemble_keys: Optional[List[str]] = None,
        chunk_size: int = 5000,
        header_batch_size: int = 10000,
        progress_callback=None
    ) -> Dict:
        """
        OPTIMIZED SINGLE-PASS IMPORT: Save traces, headers, and detect ensembles in one pass.

        This method reads the SEG-Y file ONCE and simultaneously:
        1. Writes traces to Zarr (compressed)
        2. Batches and writes headers to Parquet
        3. Detects ensemble boundaries on-the-fly
        4. Collects statistics

        Args:
            trace_generator: Generator yielding (traces_chunk, headers_chunk, start_idx, end_idx)
            n_samples: Number of samples per trace (time dimension)
            n_traces: Total number of traces in dataset
            ensemble_keys: Optional list of header fields defining ensembles (e.g., ['cdp'])
            chunk_size: Zarr chunk size for storage (default: 5000)
            header_batch_size: Number of headers to accumulate before writing (default: 10000)
            progress_callback: Optional callback function(current_trace, total_traces, phase)

        Returns:
            Dictionary with statistics:
                - compression_ratio: Zarr compression ratio
                - n_ensembles: Number of ensembles detected (0 if no ensemble_keys)
                - total_headers: Total headers saved
                - total_traces: Total traces written

        Memory Usage:
            O(max(chunk_size, header_batch_size)) - Only one chunk/batch in memory at a time

        Performance:
            3x faster than multi-pass approach (reads file once instead of 3 times)

        Example:
            >>> storage = DataStorage('output_dir')
            >>> reader = SEGYReader('file.sgy', mapping)
            >>> generator = reader.read_traces_in_chunks(chunk_size=5000)
            >>> stats = storage.save_all_streaming(
            ...     generator,
            ...     n_samples=2000,
            ...     n_traces=100000,
            ...     ensemble_keys=['cdp']
            ... )
            >>> print(f"Imported {stats['total_traces']} traces in {stats['n_ensembles']} ensembles")
        """
        print(f"  Starting optimized single-pass streaming import...")
        print(f"  Creating Zarr array: shape=({n_samples}, {n_traces})")

        # Initialize Zarr array with compression
        # No compression for maximum write speed
        z = zarr.open(
            str(self.traces_path),
            mode='w',
            shape=(n_samples, n_traces),
            chunks=(n_samples, min(chunk_size, 1000)),
            dtype=np.float32,
            compressor=None,
            zarr_format=2
        )

        # Initialize header buffer
        header_buffer = []
        trace_index_offset = 0
        total_headers = 0

        # Initialize ensemble tracking (if configured)
        ensemble_data = []
        ensemble_id = 0
        current_ensemble_start = 0
        current_ensemble_values = None

        # Statistics tracking
        total_written = 0
        uncompressed_size = 0

        print(f"  Processing data in single pass...")

        # Track last GC time for periodic cleanup
        last_gc_trace = 0

        # Single pass through the data
        for traces_chunk, headers_chunk, start_idx, end_idx in trace_generator:
            # ==========================================
            # 1. WRITE TRACES TO ZARR
            # ==========================================
            z[:, start_idx:end_idx] = traces_chunk
            total_written = end_idx
            uncompressed_size += traces_chunk.nbytes

            # ==========================================
            # 2. BATCH HEADERS FOR PARQUET
            # ==========================================
            header_buffer.extend(headers_chunk)

            # Write header batch when buffer is full
            if len(header_buffer) >= header_batch_size:
                self._write_header_batch(header_buffer, trace_index_offset, append=(total_headers > 0))
                total_headers += len(header_buffer)
                trace_index_offset += len(header_buffer)
                # Clear buffer and force list reallocation to release memory
                header_buffer = []

            # ==========================================
            # 3. DETECT ENSEMBLE BOUNDARIES (if configured)
            # ==========================================
            if ensemble_keys:
                for i, header in enumerate(headers_chunk):
                    trace_idx = start_idx + i

                    # Get current ensemble key values
                    ensemble_values = tuple(header.get(key) for key in ensemble_keys)

                    # Check if this is a new ensemble
                    if current_ensemble_values is None:
                        # First ensemble
                        current_ensemble_values = ensemble_values
                    elif ensemble_values != current_ensemble_values:
                        # Ensemble boundary detected - save previous ensemble
                        ensemble_record = {
                            'ensemble_id': ensemble_id,
                            'start_trace': current_ensemble_start,
                            'end_trace': trace_idx - 1,
                            'n_traces': trace_idx - current_ensemble_start,
                        }
                        # Add key values to record
                        for key, val in zip(ensemble_keys, current_ensemble_values):
                            ensemble_record[key] = val

                        ensemble_data.append(ensemble_record)

                        # Start new ensemble
                        ensemble_id += 1
                        current_ensemble_start = trace_idx
                        current_ensemble_values = ensemble_values

            # ==========================================
            # 4. PROGRESS CALLBACK
            # ==========================================
            if progress_callback:
                progress_callback(total_written, n_traces, "streaming")

            # Progress feedback
            if total_written % 5000 == 0 or total_written == n_traces:
                print(f"    Processed {total_written}/{n_traces} traces...")

            # ==========================================
            # 5. PERIODIC GARBAGE COLLECTION (critical for large datasets)
            # ==========================================
            if total_written - last_gc_trace >= GC_INTERVAL:
                gc.collect()
                last_gc_trace = total_written

        # ==========================================
        # FINALIZE ALL OUTPUTS
        # ==========================================

        # Write remaining headers
        if header_buffer:
            self._write_header_batch(header_buffer, trace_index_offset, append=(total_headers > 0))
            total_headers += len(header_buffer)

        # Merge all header chunk files into final headers.parquet
        self._finalize_headers()

        # Save last ensemble (if configured)
        if ensemble_keys and current_ensemble_values is not None:
            ensemble_record = {
                'ensemble_id': ensemble_id,
                'start_trace': current_ensemble_start,
                'end_trace': n_traces - 1,
                'n_traces': n_traces - current_ensemble_start,
            }
            for key, val in zip(ensemble_keys, current_ensemble_values):
                ensemble_record[key] = val
            ensemble_data.append(ensemble_record)

        # Write ensemble index to Parquet
        n_ensembles = 0
        if ensemble_data:
            df_ensembles = pd.DataFrame(ensemble_data)
            df_ensembles.to_parquet(
                self.ensemble_index_path,
                engine='pyarrow',
                compression='snappy',
                index=False
            )
            n_ensembles = len(ensemble_data)
            print(f"    Saved {n_ensembles} ensembles")

        # Calculate compression ratio
        compressed_size = sum(f.stat().st_size for f in self.traces_path.rglob('*') if f.is_file())
        compression_ratio = uncompressed_size / compressed_size if compressed_size > 0 else 1.0

        print(f"  ✓ Single-pass import complete!")
        print(f"    Zarr compression: {compression_ratio:.2f}x")
        print(f"    Headers saved: {total_headers}")
        print(f"    Ensembles detected: {n_ensembles}")

        return {
            'compression_ratio': compression_ratio,
            'n_ensembles': n_ensembles,
            'total_headers': total_headers,
            'total_traces': total_written,
            'uncompressed_size_mb': uncompressed_size / 1024 / 1024,
            'compressed_size_mb': compressed_size / 1024 / 1024
        }

    def detect_ensembles_streaming(self, header_generator, ensemble_keys: List[str]):
        """
        Detect ensemble boundaries on-the-fly from streaming headers.

        This method processes headers as they stream and detects when ensemble key values
        change, yielding ensemble boundary information without accumulating all headers.

        Args:
            header_generator: Generator yielding (traces_chunk, headers_chunk, start_idx, end_idx)
            ensemble_keys: List of header field names that define ensemble boundaries
                          (e.g., ['cdp'] or ['inline', 'crossline'])

        Yields:
            Tuple containing:
                - ensemble_id: Sequential ensemble identifier (starting from 0)
                - start_trace: Starting trace index for this ensemble
                - end_trace: Ending trace index for this ensemble (inclusive)
                - key_values: Dict of ensemble key values for this ensemble

        Memory Usage:
            O(1) - Only tracks current ensemble state

        Example:
            >>> storage = DataStorage('output_dir')
            >>> reader = SEGYReader('file.sgy', mapping)
            >>> generator = reader.read_traces_in_chunks(chunk_size=5000)
            >>> for ens_id, start, end, keys in storage.detect_ensembles_streaming(generator, ['cdp']):
            ...     print(f"Ensemble {ens_id}: traces {start}-{end}, CDP={keys['cdp']}")
        """
        if not ensemble_keys:
            raise ValueError("ensemble_keys cannot be empty")

        ensemble_id = 0
        current_ensemble_start = 0
        current_ensemble_values = None
        current_trace_idx = 0

        print(f"  Detecting ensembles on-the-fly (keys: {ensemble_keys})...")

        # Process headers from generator
        for traces_chunk, headers_chunk, start_idx, end_idx in header_generator:
            for i, header in enumerate(headers_chunk):
                # Get current ensemble key values
                ensemble_values = tuple(header.get(key) for key in ensemble_keys)

                # Check if this is a new ensemble
                if current_ensemble_values is None:
                    # First ensemble
                    current_ensemble_values = ensemble_values
                elif ensemble_values != current_ensemble_values:
                    # Ensemble boundary detected - yield previous ensemble
                    key_dict = dict(zip(ensemble_keys, current_ensemble_values))
                    yield ensemble_id, current_ensemble_start, current_trace_idx - 1, key_dict

                    # Start new ensemble
                    ensemble_id += 1
                    current_ensemble_start = current_trace_idx
                    current_ensemble_values = ensemble_values

                current_trace_idx += 1

            # Progress feedback
            if current_trace_idx % 5000 == 0:
                print(f"    Processed {current_trace_idx} traces, found {ensemble_id + 1} ensembles...")

        # Yield final ensemble
        if current_ensemble_values is not None:
            key_dict = dict(zip(ensemble_keys, current_ensemble_values))
            yield ensemble_id, current_ensemble_start, current_trace_idx - 1, key_dict

        print(f"    Completed: {ensemble_id + 1} ensembles detected from {current_trace_idx} traces")

    def save_ensemble_index_streaming(self, ensemble_generator):
        """
        Save ensemble boundaries from streaming generator to Parquet.

        Args:
            ensemble_generator: Generator yielding (ensemble_id, start, end, key_values)

        Returns:
            Total number of ensembles saved
        """
        print(f"  Saving ensemble index to Parquet...")

        ensemble_data = []
        for ensemble_id, start_trace, end_trace, key_values in ensemble_generator:
            ensemble_record = {
                'ensemble_id': ensemble_id,
                'start_trace': start_trace,
                'end_trace': end_trace,
                'n_traces': end_trace - start_trace + 1,
            }
            # Add key values to record
            ensemble_record.update(key_values)
            ensemble_data.append(ensemble_record)

        # Save to Parquet
        if ensemble_data:
            df_ensembles = pd.DataFrame(ensemble_data)
            df_ensembles.to_parquet(
                self.ensemble_index_path,
                engine='pyarrow',
                compression='snappy',
                index=False
            )
            print(f"    Saved {len(ensemble_data)} ensembles")
            return len(ensemble_data)
        else:
            print("    No ensembles to save")
            return 0

    def _save_headers_parquet(self, headers: List[Dict]):
        """Save headers to Parquet."""
        print(f"  Saving headers to Parquet: {len(headers)} traces")

        # Convert to DataFrame
        df_headers = pd.DataFrame(headers)

        # Add trace index column
        df_headers['trace_index'] = np.arange(len(headers))

        # Save to Parquet with compression
        df_headers.to_parquet(
            self.headers_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )

        print(f"    Header columns: {list(df_headers.columns)}")

    def _save_ensemble_index(self, ensembles: List[Tuple[int, int]], n_traces: int = None):
        """
        Save ensemble boundaries to Parquet.

        Always creates an ensemble_index.parquet file. If no ensembles are provided,
        creates a single ensemble covering all traces (required for parallel processing).
        """
        if not ensembles:
            # No ensembles specified - create single ensemble for all traces
            if n_traces is None:
                # Try to get n_traces from trace index or zarr
                if self.trace_index_path.exists():
                    trace_idx = pd.read_parquet(self.trace_index_path)
                    n_traces = len(trace_idx)
                elif self.traces_path.exists():
                    import zarr
                    z = zarr.open(self.traces_path, mode='r')
                    n_traces = z.shape[1] if len(z.shape) > 1 else z.shape[0]
                else:
                    print("    Warning: Cannot determine n_traces for default ensemble index")
                    return

            print(f"  Saving ensemble index: 1 ensemble (default, {n_traces:,} traces)")
            ensemble_data = [{
                'ensemble_id': 0,
                'ensemble_value': 0,
                'start_trace': 0,
                'end_trace': n_traces - 1,
                'n_traces': n_traces
            }]
        else:
            print(f"  Saving ensemble index: {len(ensembles)} ensembles")
            ensemble_data = []
            for i, (start, end) in enumerate(ensembles):
                ensemble_data.append({
                    'ensemble_id': i,
                    'start_trace': start,
                    'end_trace': end,
                    'n_traces': end - start + 1
                })

        df_ensembles = pd.DataFrame(ensemble_data)
        df_ensembles.to_parquet(
            self.ensemble_index_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )

    def ensure_ensemble_index(self, n_traces: int = None) -> bool:
        """
        Ensure ensemble_index.parquet exists, creating default if missing.

        This is useful for datasets imported before ensemble index was mandatory.

        Args:
            n_traces: Total number of traces (auto-detected if not provided)

        Returns:
            True if index exists or was created, False on failure
        """
        if self.ensemble_index_path.exists():
            return True

        print(f"  Creating missing ensemble_index.parquet...")

        # Determine n_traces
        if n_traces is None:
            if self.trace_index_path.exists():
                trace_idx = pd.read_parquet(self.trace_index_path)
                n_traces = len(trace_idx)
            elif self.traces_path.exists():
                import zarr
                z = zarr.open(self.traces_path, mode='r')
                n_traces = z.shape[1] if len(z.shape) > 1 else z.shape[0]
            else:
                print("    Error: Cannot determine n_traces")
                return False

        # Create default single-ensemble index
        ensemble_data = [{
            'ensemble_id': 0,
            'ensemble_value': 0,
            'start_trace': 0,
            'end_trace': n_traces - 1,
            'n_traces': n_traces
        }]

        df_ensembles = pd.DataFrame(ensemble_data)
        df_ensembles.to_parquet(
            self.ensemble_index_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        print(f"    Created default ensemble index ({n_traces:,} traces)")
        return True

    def _save_trace_index(self, n_traces: int):
        """Save trace index for fast lookup."""
        print(f"  Saving trace index: {n_traces} traces")

        # Simple trace index (can be enhanced with spatial indices later)
        df_index = pd.DataFrame({
            'trace_index': np.arange(n_traces),
            'global_trace_id': np.arange(n_traces)
        })

        df_index.to_parquet(
            self.trace_index_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )

    def _save_metadata(self, seismic_data: SeismicData):
        """Save metadata to JSON."""
        print(f"  Saving metadata")

        metadata = {
            'shape': list(seismic_data.traces.shape),
            'sample_rate': seismic_data.sample_rate,
            'n_samples': seismic_data.n_samples,
            'n_traces': seismic_data.n_traces,
            'duration_ms': seismic_data.duration,
            'nyquist_freq': seismic_data.nyquist_freq,
            'seismic_metadata': seismic_data.metadata,
            'storage_info': {
                'zarr_chunks': f"({seismic_data.n_samples}, chunk_size)",
                'parquet_compression': 'snappy',
                'zarr_compression': 'none'
            }
        }

        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def load_seismic_data(self) -> Tuple[SeismicData, pd.DataFrame, pd.DataFrame]:
        """
        Load seismic data from storage.

        Returns:
            Tuple of:
                - SeismicData object
                - Headers DataFrame
                - Ensemble index DataFrame
        """
        print(f"Loading data from: {self.output_dir}")

        # Load metadata
        with open(self.metadata_path, 'r') as f:
            metadata = json.load(f)

        # Load traces from Zarr
        print(f"  Loading traces from Zarr...")
        z = zarr.open(str(self.traces_path), mode='r')
        traces = np.array(z)

        # Load headers
        print(f"  Loading headers from Parquet...")
        df_headers = pd.read_parquet(self.headers_path)

        # Load ensemble index
        df_ensembles = None
        if self.ensemble_index_path.exists():
            print(f"  Loading ensemble index...")
            df_ensembles = pd.read_parquet(self.ensemble_index_path)

        # Create SeismicData object
        seismic_data = SeismicData(
            traces=traces,
            sample_rate=metadata['sample_rate'],
            metadata=metadata.get('seismic_metadata', {})
        )

        print(f"✓ Data loaded: {seismic_data}")

        return seismic_data, df_headers, df_ensembles

    def get_ensemble_traces(self, ensemble_id: int) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Get traces and headers for a specific ensemble.

        Args:
            ensemble_id: Ensemble ID to retrieve

        Returns:
            Tuple of (traces array, headers DataFrame)
        """
        # Load ensemble index
        df_ensembles = pd.read_parquet(self.ensemble_index_path)

        # Get ensemble boundaries
        ensemble = df_ensembles[df_ensembles['ensemble_id'] == ensemble_id].iloc[0]
        start = int(ensemble['start_trace'])
        end = int(ensemble['end_trace'])

        # Load traces for this ensemble
        z = zarr.open(str(self.traces_path), mode='r')
        traces = z[:, start:end+1]

        # Load headers for this ensemble
        df_headers = pd.read_parquet(self.headers_path)
        ensemble_headers = df_headers[(df_headers['trace_index'] >= start) &
                                      (df_headers['trace_index'] <= end)]

        return traces, ensemble_headers

    def query_headers(self, query: str) -> pd.DataFrame:
        """
        Query headers using pandas query syntax.

        Args:
            query: Pandas query string (e.g., "cdp > 1000 and offset < 2000")

        Returns:
            Filtered DataFrame
        """
        df_headers = pd.read_parquet(self.headers_path)
        return df_headers.query(query)

    def get_ensemble_index(self) -> Optional[pd.DataFrame]:
        """
        Load ensemble index without loading full dataset.

        Returns:
            DataFrame with ensemble boundaries, or None if not available
        """
        if self.ensemble_index_path.exists():
            return pd.read_parquet(self.ensemble_index_path)
        return None

    def get_statistics(self) -> Dict:
        """Get storage statistics without loading full data."""
        stats = {}

        # Load metadata first (lightweight)
        metadata = None
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)

        if self.traces_path.exists():
            # Zarr stats
            z = zarr.open(str(self.traces_path), mode='r')
            stats['zarr'] = {
                'shape': z.shape,
                'dtype': str(z.dtype),
                'chunks': z.chunks,
                'size_mb': sum(f.stat().st_size for f in self.traces_path.rglob('*') if f.is_file()) / 1024 / 1024
            }

        if self.headers_path.exists():
            # Parquet stats - use metadata instead of loading full file
            # Only load parquet metadata (no actual data)
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(self.headers_path)

            stats['headers'] = {
                'n_traces': metadata.get('n_traces', parquet_file.metadata.num_rows) if metadata else parquet_file.metadata.num_rows,
                'n_columns': len(parquet_file.schema),
                'columns': [field.name for field in parquet_file.schema],
                'size_mb': self.headers_path.stat().st_size / 1024 / 1024
            }

        if self.ensemble_index_path.exists():
            df_ensembles = pd.read_parquet(self.ensemble_index_path)
            stats['ensembles'] = {
                'n_ensembles': len(df_ensembles),
                'avg_traces_per_ensemble': df_ensembles['n_traces'].mean()
            }

        return stats

    def __repr__(self) -> str:
        return f"DataStorage('{self.output_dir}')"
