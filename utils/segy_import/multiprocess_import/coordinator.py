"""
Coordinator for parallel multiprocess SEGY import.

Orchestrates the full import pipeline:
1. Pre-create shared Zarr array
2. Partition file into segments
3. Launch parallel workers (write directly to shared Zarr)
4. Merge header parquet files
"""

import os
import time
import segyio
import zarr
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Any
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from .partitioner import SmartPartitioner, PartitionConfig, Segment, get_ensemble_byte_location
from .worker import WorkerTask, WorkerResult, import_segment


@dataclass
class ImportConfig:
    """Configuration for parallel import."""
    segy_path: str
    output_dir: str
    header_mapping: Any  # HeaderMapping object
    ensemble_key: Optional[str] = 'cdp'
    n_workers: Optional[int] = None  # Auto-detect if None
    chunk_size: int = 10000


@dataclass
class ImportProgress:
    """Progress information for callbacks."""
    phase: str                # 'partitioning', 'importing', 'merging'
    current_traces: int
    total_traces: int
    active_workers: int
    worker_progress: Dict[int, int] = field(default_factory=dict)
    elapsed_time: float = 0.0
    eta_seconds: float = 0.0


@dataclass
class ImportResult:
    """Final result of import operation."""
    success: bool
    output_dir: str
    traces_path: str
    headers_path: str
    n_traces: int
    n_segments: int
    elapsed_time: float
    error: Optional[str] = None


def get_optimal_workers() -> int:
    """Get optimal number of worker processes."""
    cpu_count = os.cpu_count() or 4
    # Leave 2 cores free for system/UI, minimum 2 workers
    return max(2, cpu_count - 2)


class ParallelImportCoordinator:
    """
    Orchestrates parallel multiprocess SEGY import.

    Workers write directly to a pre-created shared Zarr array,
    eliminating the need for trace data merging.

    Usage:
        config = ImportConfig(
            segy_path='file.sgy',
            output_dir='output/',
            header_mapping=mapping
        )
        coordinator = ParallelImportCoordinator(config)
        result = coordinator.run(progress_callback=update_ui)
    """

    def __init__(self, config: ImportConfig):
        """
        Initialize coordinator.

        Args:
            config: Import configuration
        """
        self.config = config
        self.n_workers = config.n_workers or get_optimal_workers()
        self._cancel_requested = False
        self._manager = None
        self._progress_queue = None

    def run(self, progress_callback: Optional[Callable[[ImportProgress], None]] = None) -> ImportResult:
        """
        Run the full parallel import pipeline.

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            ImportResult with outcome
        """
        start_time = time.time()
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Phase 1: Get file info
            file_info = self._get_file_info()
            n_traces = file_info['n_traces']
            n_samples = file_info['n_samples']

            if progress_callback:
                progress_callback(ImportProgress(
                    phase='partitioning',
                    current_traces=0,
                    total_traces=n_traces,
                    active_workers=0
                ))

            # Phase 2: Partition into segments
            segments = self._partition(n_traces)
            print(f"  Partitioned into {len(segments)} segments")

            # Phase 3: Pre-create shared Zarr array
            traces_path = self._create_shared_zarr(output_dir, n_samples, n_traces)
            print(f"  Created shared Zarr array: {n_samples} x {n_traces:,}")

            # Phase 4: Run parallel workers
            if progress_callback:
                progress_callback(ImportProgress(
                    phase='importing',
                    current_traces=0,
                    total_traces=n_traces,
                    active_workers=len(segments)
                ))

            worker_results = self._run_workers(
                segments, n_samples, n_traces, progress_callback
            )

            # Check for failures
            failed = [r for r in worker_results if not r.success]
            if failed:
                errors = "; ".join([f"Segment {r.segment_id}: {r.error}" for r in failed])
                raise RuntimeError(f"Worker failures: {errors}")

            # Phase 5: Merge header files only (traces already in shared Zarr)
            if progress_callback:
                progress_callback(ImportProgress(
                    phase='merging',
                    current_traces=n_traces,
                    total_traces=n_traces,
                    active_workers=0
                ))

            headers_path = self._merge_headers(output_dir, worker_results)
            self._create_indices(output_dir, n_traces)
            self._build_ensemble_index(output_dir, headers_path)
            self._cleanup_segment_files(output_dir, worker_results)

            # Save metadata
            self._save_metadata(output_dir, file_info, len(segments))

            elapsed = time.time() - start_time

            return ImportResult(
                success=True,
                output_dir=str(output_dir),
                traces_path=traces_path,
                headers_path=headers_path,
                n_traces=n_traces,
                n_segments=len(segments),
                elapsed_time=elapsed
            )

        except Exception as e:
            import traceback
            elapsed = time.time() - start_time
            return ImportResult(
                success=False,
                output_dir=str(output_dir),
                traces_path="",
                headers_path="",
                n_traces=0,
                n_segments=0,
                elapsed_time=elapsed,
                error=f"{str(e)}\n{traceback.format_exc()}"
            )

    def _get_file_info(self) -> Dict[str, Any]:
        """Get basic file information."""
        with segyio.open(self.config.segy_path, 'r', ignore_geometry=True) as f:
            return {
                'n_traces': f.tracecount,
                'n_samples': len(f.samples),
                'sample_interval': segyio.tools.dt(f) / 1000.0
            }

    def _partition(self, n_traces: int) -> List[Segment]:
        """Partition file into segments."""
        ensemble_byte = 21  # Default CDP
        if self.config.ensemble_key:
            ensemble_byte = get_ensemble_byte_location(self.config.ensemble_key)

        partition_config = PartitionConfig(
            segy_path=self.config.segy_path,
            n_segments=self.n_workers,
            total_traces=n_traces,
            ensemble_key=self.config.ensemble_key,
            ensemble_byte=ensemble_byte
        )

        partitioner = SmartPartitioner(partition_config)
        segments = partitioner.partition()

        # Log partition stats
        stats = partitioner.get_partition_stats(segments)
        print(f"  Partition stats: {stats}")

        return segments

    def _create_shared_zarr(self, output_dir: Path, n_samples: int, n_traces: int) -> str:
        """
        Pre-create shared Zarr array for all workers to write to.

        Args:
            output_dir: Output directory
            n_samples: Samples per trace
            n_traces: Total number of traces

        Returns:
            Path to traces.zarr
        """
        traces_path = output_dir / "traces.zarr"

        # Create Zarr array (no compression for speed)
        zarr.open(
            str(traces_path),
            mode='w',
            shape=(n_samples, n_traces),
            chunks=(n_samples, 1000),
            dtype=np.float32,
            compressor=None,
            zarr_format=2
        )

        return str(traces_path)

    def _run_workers(
        self,
        segments: List[Segment],
        n_samples: int,
        n_traces: int,
        progress_callback: Optional[Callable]
    ) -> List[WorkerResult]:
        """
        Run worker processes in parallel.

        Args:
            segments: List of segments to process
            n_samples: Samples per trace
            n_traces: Total traces (for progress)
            progress_callback: Progress callback

        Returns:
            List of worker results
        """
        # Serialize header mapping for workers
        header_mapping_dict = self.config.header_mapping.get_all_mappings()

        # Create worker tasks
        tasks = []
        for segment in segments:
            task = WorkerTask(
                segment_id=segment.segment_id,
                segy_path=self.config.segy_path,
                output_dir=str(self.config.output_dir),
                start_trace=segment.start_trace,
                end_trace=segment.end_trace,
                n_samples=n_samples,
                header_mapping_dict=header_mapping_dict,
                chunk_size=self.config.chunk_size
            )
            tasks.append(task)

        # Use multiprocessing Manager for progress queue
        manager = mp.Manager()
        progress_queue = manager.Queue()

        # Track progress per worker
        worker_progress = {s.segment_id: 0 for s in segments}
        results = []
        processed_futures = set()  # Track which futures we've already processed

        print(f"  Launching {len(tasks)} worker processes...")

        # Submit all tasks
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit tasks
            futures = {
                executor.submit(import_segment, task, progress_queue): task
                for task in tasks
            }

            # Monitor progress while waiting
            start_time = time.time()

            while len(processed_futures) < len(futures):
                # Check for completed futures
                for future in futures:
                    if future.done() and future not in processed_futures:
                        processed_futures.add(future)
                        try:
                            result = future.result(timeout=0.1)
                            results.append(result)
                            worker_progress[result.segment_id] = result.n_traces
                            print(f"    Worker {result.segment_id} completed: "
                                  f"{result.n_traces:,} traces in {result.elapsed_time:.1f}s")
                        except Exception as e:
                            task = futures[future]
                            results.append(WorkerResult(
                                segment_id=task.segment_id,
                                headers_path="",
                                n_traces=0,
                                start_trace=task.start_trace,
                                elapsed_time=0,
                                success=False,
                                error=str(e)
                            ))

                # Drain progress queue
                while not progress_queue.empty():
                    try:
                        segment_id, traces_done = progress_queue.get_nowait()
                        worker_progress[segment_id] = traces_done
                    except:
                        break

                # Update progress callback
                if progress_callback:
                    total_done = sum(worker_progress.values())
                    elapsed = time.time() - start_time
                    rate = total_done / elapsed if elapsed > 0 else 0
                    eta = (n_traces - total_done) / rate if rate > 0 else 0

                    progress_callback(ImportProgress(
                        phase='importing',
                        current_traces=total_done,
                        total_traces=n_traces,
                        active_workers=len(futures) - len(processed_futures),
                        worker_progress=worker_progress.copy(),
                        elapsed_time=elapsed,
                        eta_seconds=eta
                    ))

                # Small sleep to avoid busy-waiting
                time.sleep(0.1)

        return results

    def _merge_headers(self, output_dir: Path, results: List[WorkerResult]) -> str:
        """
        Merge segment header parquet files into single file.

        Headers already have correct global trace indices.

        Args:
            output_dir: Output directory
            results: Worker results with header file paths

        Returns:
            Path to merged headers.parquet
        """
        print(f"    Merging headers from {len(results)} segments...")

        final_path = output_dir / "headers.parquet"

        # Sort by segment_id to ensure correct order
        sorted_results = sorted(results, key=lambda r: r.segment_id)

        all_dfs = []
        for result in sorted_results:
            if result.headers_path and Path(result.headers_path).exists():
                df = pd.read_parquet(result.headers_path)
                all_dfs.append(df)
                print(f"      Segment {result.segment_id}: {len(df):,} headers")

        # Concatenate all DataFrames
        final_df = pd.concat(all_dfs, ignore_index=True)

        # Sort by trace_index to ensure correct order
        final_df = final_df.sort_values('trace_index').reset_index(drop=True)

        # Write final Parquet
        final_df.to_parquet(
            final_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )

        print(f"      Total: {len(final_df):,} headers")

        return str(final_path)

    def _create_indices(self, output_dir: Path, n_traces: int):
        """Create trace index file."""
        df_index = pd.DataFrame({
            'trace_index': np.arange(n_traces),
            'global_trace_id': np.arange(n_traces)
        })

        index_path = output_dir / "trace_index.parquet"
        df_index.to_parquet(
            index_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )

    def _build_ensemble_index(self, output_dir: Path, headers_path: str):
        """
        Build ensemble index from headers based on ensemble key.

        This creates the ensemble_index.parquet file needed for gather navigation
        and parallel processing. Always creates an index even if no key is specified
        (treats entire dataset as single ensemble).
        """
        # Load headers
        headers_df = pd.read_parquet(headers_path)
        n_traces = len(headers_df)

        # If no ensemble key, create single-ensemble index covering all traces
        if not self.config.ensemble_key:
            print(f"    Building default ensemble index (single ensemble for all {n_traces:,} traces)...")
            ensembles = [{
                'ensemble_id': 0,
                'ensemble_value': 0,
                'start_trace': 0,
                'end_trace': n_traces - 1,
                'n_traces': n_traces
            }]
        else:
            print(f"    Building ensemble index on '{self.config.ensemble_key}'...")

            # Check if ensemble key column exists
            if self.config.ensemble_key not in headers_df.columns:
                print(f"      Warning: Ensemble key '{self.config.ensemble_key}' not found in headers")
                print(f"      Creating default single-ensemble index instead...")
                ensembles = [{
                    'ensemble_id': 0,
                    'ensemble_value': 0,
                    'start_trace': 0,
                    'end_trace': n_traces - 1,
                    'n_traces': n_traces
                }]
            else:
                # Build ensemble index from key
                ensembles = []
                ensemble_col = headers_df[self.config.ensemble_key].values

                # Find ensemble boundaries
                current_value = ensemble_col[0]
                start_trace = 0
                ensemble_id = 0

                for i in range(1, len(ensemble_col)):
                    if ensemble_col[i] != current_value:
                        # End of ensemble
                        ensembles.append({
                            'ensemble_id': ensemble_id,
                            'ensemble_value': int(current_value),
                            'start_trace': start_trace,
                            'end_trace': i - 1,
                            'n_traces': i - start_trace
                        })
                        ensemble_id += 1
                        current_value = ensemble_col[i]
                        start_trace = i

                # Last ensemble
                ensembles.append({
                    'ensemble_id': ensemble_id,
                    'ensemble_value': int(current_value),
                    'start_trace': start_trace,
                    'end_trace': len(ensemble_col) - 1,
                    'n_traces': len(ensemble_col) - start_trace
                })

        # Save to parquet
        ensemble_df = pd.DataFrame(ensembles)
        ensemble_path = output_dir / 'ensemble_index.parquet'
        ensemble_df.to_parquet(
            ensemble_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )

        print(f"      Created {len(ensembles):,} ensembles")

    def _cleanup_segment_files(self, output_dir: Path, results: List[WorkerResult]):
        """Remove temporary segment header files."""
        print(f"    Cleaning up segment files...")

        for result in results:
            if result.headers_path:
                header_path = Path(result.headers_path)
                try:
                    if header_path.exists():
                        header_path.unlink()
                except Exception as e:
                    print(f"      Warning: Could not remove {header_path}: {e}")

    def _save_metadata(self, output_dir: Path, file_info: dict, n_segments: int):
        """Save import metadata."""
        import json

        metadata = {
            'shape': [file_info['n_samples'], file_info['n_traces']],
            'sample_rate': file_info['sample_interval'],
            'n_samples': file_info['n_samples'],
            'n_traces': file_info['n_traces'],
            'duration_ms': (file_info['n_samples'] - 1) * file_info['sample_interval'],
            'nyquist_freq': 1000.0 / (2.0 * file_info['sample_interval']),
            'import_info': {
                'method': 'parallel_multiprocess',
                'n_workers': self.n_workers,
                'n_segments': n_segments
            },
            'storage_info': {
                'zarr_compression': 'none',
                'parquet_compression': 'snappy'
            }
        }

        metadata_path = output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def cancel(self):
        """Request cancellation of import."""
        self._cancel_requested = True
