"""
Gather navigator - manages ensemble/gather navigation and data extraction.
Enables viewing one gather at a time from multi-gather seismic data.
"""
import numpy as np
import pandas as pd
from PyQt6.QtCore import QObject, pyqtSignal
from typing import Optional, Tuple, Dict
from collections import OrderedDict
import threading
import time
import sys
from models.seismic_data import SeismicData
from models.lazy_seismic_data import LazySeismicData


class GatherNavigator(QObject):
    """
    Manages navigation through seismic gathers/ensembles.

    Tracks current gather, provides next/previous navigation,
    and extracts gather-specific data from full dataset.

    Signals:
        gather_changed: Emitted when current gather changes (gather_id, gather_info)
        navigation_state_changed: Emitted when navigation state changes (can_prev, can_next)
    """

    gather_changed = pyqtSignal(int, dict)  # gather_id, gather_info
    navigation_state_changed = pyqtSignal(bool, bool)  # can_prev, can_next
    sort_keys_changed = pyqtSignal(list)  # sort_keys

    def __init__(self):
        super().__init__()

        # Full dataset (legacy mode)
        self.full_data: Optional[SeismicData] = None
        self.headers_df: Optional[pd.DataFrame] = None
        self.ensembles_df: Optional[pd.DataFrame] = None

        # Lazy loading mode
        self.lazy_data: Optional[LazySeismicData] = None
        self._ensemble_cache: OrderedDict = OrderedDict()  # LRU cache: gather_id -> (SeismicData, headers_df)
        self._max_cached_ensembles = 5
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_lock = threading.RLock()  # Thread-safe cache access

        # Background prefetching
        self._prefetch_thread: Optional[threading.Thread] = None
        self._prefetch_event = threading.Event()  # Signal to trigger prefetch
        self._stop_prefetch = threading.Event()  # Signal to stop thread
        self._prefetch_gather_id: Optional[int] = None  # Target gather for prefetching

        # Current state
        self.current_gather_id: int = 0
        self.n_gathers: int = 0

        # Ensemble configuration
        self.ensemble_keys: list = []

        # In-gather sorting
        self.sort_keys: list = []  # List of header names to sort by within each gather

    def load_data(self, seismic_data: SeismicData, headers_df: pd.DataFrame,
                  ensembles_df: Optional[pd.DataFrame] = None):
        """
        Load full seismic dataset with ensemble information.

        Args:
            seismic_data: Full seismic data
            headers_df: DataFrame with trace headers
            ensembles_df: DataFrame with ensemble boundaries (optional)
        """
        self.full_data = seismic_data
        self.headers_df = headers_df
        self.ensembles_df = ensembles_df

        if ensembles_df is not None and len(ensembles_df) > 0:
            self.n_gathers = len(ensembles_df)
            self.current_gather_id = 0

            # Extract ensemble keys from metadata if available
            if 'header_mapping' in seismic_data.metadata:
                mapping = seismic_data.metadata['header_mapping']
                self.ensemble_keys = mapping.get('ensemble_keys', [])
        else:
            # No ensembles - treat as single gather
            self.n_gathers = 1
            self.current_gather_id = 0
            self.ensemble_keys = []

        # Emit initial state
        self._emit_state_changes()

    def load_lazy_data(self, lazy_data: LazySeismicData,
                       ensembles_df: Optional[pd.DataFrame] = None):
        """
        Load lazy seismic dataset with ensemble information.

        Args:
            lazy_data: LazySeismicData for memory-efficient loading
            ensembles_df: DataFrame with ensemble boundaries (optional)
        """
        self.lazy_data = lazy_data
        self.full_data = None  # Clear full data to prevent conflicts
        self.headers_df = None  # Headers loaded on-demand from lazy_data
        self.ensembles_df = ensembles_df
        self._ensemble_cache.clear()  # Clear cache when new data loaded
        self._cache_hits = 0
        self._cache_misses = 0

        if ensembles_df is not None and len(ensembles_df) > 0:
            self.n_gathers = len(ensembles_df)
            self.current_gather_id = 0

            # Extract ensemble keys from metadata if available
            if 'header_mapping' in lazy_data.metadata:
                mapping = lazy_data.metadata['header_mapping']
                self.ensemble_keys = mapping.get('ensemble_keys', [])
        else:
            # No ensembles - treat as single gather
            self.n_gathers = 1
            self.current_gather_id = 0
            self.ensemble_keys = []

        # Start prefetch thread for lazy loading
        self._start_prefetch_thread()

        # Emit initial state
        self._emit_state_changes()

    def has_gathers(self) -> bool:
        """Check if data has multiple gathers."""
        return self.n_gathers > 1

    def set_sort_keys(self, sort_keys: list):
        """
        Set header keys for in-gather sorting.

        Args:
            sort_keys: List of header names to sort by (in order of priority)
        """
        self.sort_keys = sort_keys if sort_keys else []
        self.sort_keys_changed.emit(self.sort_keys)

        # Re-emit gather changed to trigger re-display with new sort
        if self.full_data is not None or self.lazy_data is not None:
            gather_info = self._get_gather_info(self.current_gather_id)
            self.gather_changed.emit(self.current_gather_id, gather_info)

    def get_available_sort_headers(self) -> list:
        """
        Get list of available header names that can be used for sorting.

        Returns:
            List of header column names
        """
        if self.lazy_data is not None:
            # Lazy mode - load just first header to get column names
            try:
                first_header = self.lazy_data.get_headers([0])
                return list(first_header.columns)
            except Exception as e:
                print(f"Warning: Failed to load headers: {e}")
                return []
        elif self.headers_df is not None:
            # Full data mode
            return list(self.headers_df.columns)
        else:
            return []

    def get_current_gather(self) -> Tuple[SeismicData, pd.DataFrame, Dict]:
        """
        Get data for current gather.

        Returns:
            Tuple of:
                - SeismicData for current gather (sorted if sort_keys set)
                - Headers DataFrame for current gather (sorted if sort_keys set)
                - Gather info dictionary
        """
        # Check if using lazy loading mode
        if self.lazy_data is not None:
            return self._get_lazy_gather(self.current_gather_id)

        # Legacy full data mode
        if self.full_data is None:
            raise ValueError("No data loaded")

        if not self.has_gathers():
            # Return full data if no gathers (with sorting if enabled)
            gather_data = self.full_data
            gather_headers = self.headers_df.copy()

            # Apply sorting if sort keys are set
            if self.sort_keys:
                gather_data, gather_headers = self._sort_gather_traces(
                    gather_data, gather_headers
                )

            return gather_data, gather_headers, self._get_gather_info(0)

        # Get ensemble boundaries
        ensemble = self.ensembles_df.iloc[self.current_gather_id]
        start_trace = int(ensemble['start_trace'])
        end_trace = int(ensemble['end_trace'])
        n_traces = int(ensemble['n_traces'])

        # Extract traces for this gather
        gather_traces = self.full_data.traces[:, start_trace:end_trace+1]

        # Create SeismicData for this gather
        gather_data = SeismicData(
            traces=gather_traces,
            sample_rate=self.full_data.sample_rate,
            metadata={
                **self.full_data.metadata,
                'gather_id': self.current_gather_id,
                'start_trace': start_trace,
                'end_trace': end_trace,
                'n_traces': n_traces
            }
        )

        # Extract headers for this gather
        gather_headers = self.headers_df.iloc[start_trace:end_trace+1].copy()
        gather_headers = gather_headers.reset_index(drop=True)

        # Apply sorting if sort keys are set
        if self.sort_keys:
            gather_data, gather_headers = self._sort_gather_traces(
                gather_data, gather_headers
            )

        # Get gather info
        gather_info = self._get_gather_info(self.current_gather_id)

        return gather_data, gather_headers, gather_info

    def _get_lazy_gather(self, gather_id: int) -> Tuple[SeismicData, pd.DataFrame, Dict]:
        """
        Get gather data using lazy loading with LRU caching (thread-safe).

        Args:
            gather_id: Gather ID to load

        Returns:
            Tuple of (gather_data, gather_headers, gather_info)
        """
        # Check cache first (thread-safe)
        with self._cache_lock:
            if gather_id in self._ensemble_cache:
                self._cache_hits += 1
                # Move to end (mark as most recently used)
                self._ensemble_cache.move_to_end(gather_id)
                gather_data, gather_headers = self._ensemble_cache[gather_id]

                # Apply sorting if sort keys changed since caching
                if self.sort_keys:
                    gather_data, gather_headers = self._sort_gather_traces(
                        gather_data, gather_headers
                    )

                gather_info = self._get_gather_info(gather_id)
                return gather_data, gather_headers, gather_info

        # Cache miss - load from Zarr (outside lock to avoid blocking other threads)
        with self._cache_lock:
            self._cache_misses += 1

        if not self.has_gathers():
            # Load all data for single gather mode
            gather_traces = self.lazy_data.get_trace_range(0, self.lazy_data.n_traces)

            gather_data = SeismicData(
                traces=gather_traces,
                sample_rate=self.lazy_data.sample_rate,
                metadata={
                    **self.lazy_data.metadata,
                    'gather_id': 0,
                    'n_traces': self.lazy_data.n_traces
                }
            )

            # Load headers for all traces
            trace_indices = list(range(self.lazy_data.n_traces))
            gather_headers = self.lazy_data.get_headers(trace_indices)
        else:
            # Load specific ensemble
            ensemble = self.ensembles_df.iloc[gather_id]
            ensemble_id = int(ensemble['ensemble_id'])
            start_trace = int(ensemble['start_trace'])
            end_trace = int(ensemble['end_trace'])
            n_traces = int(ensemble['n_traces'])

            # Load ensemble traces from Zarr
            gather_traces = self.lazy_data.get_ensemble(ensemble_id)

            # Create SeismicData for this gather
            gather_data = SeismicData(
                traces=gather_traces,
                sample_rate=self.lazy_data.sample_rate,
                metadata={
                    **self.lazy_data.metadata,
                    'gather_id': gather_id,
                    'ensemble_id': ensemble_id,
                    'start_trace': start_trace,
                    'end_trace': end_trace,
                    'n_traces': n_traces
                }
            )

            # Load headers for this gather
            trace_indices = list(range(start_trace, end_trace + 1))
            gather_headers = self.lazy_data.get_headers(trace_indices)
            gather_headers = gather_headers.reset_index(drop=True)

        # Add to cache with LRU eviction
        self._add_to_cache(gather_id, gather_data, gather_headers)

        # Apply sorting if sort keys are set
        if self.sort_keys:
            gather_data, gather_headers = self._sort_gather_traces(
                gather_data, gather_headers
            )

        # Get gather info
        gather_info = self._get_gather_info(gather_id)

        return gather_data, gather_headers, gather_info

    def _add_to_cache(self, gather_id: int, gather_data: SeismicData,
                      gather_headers: pd.DataFrame):
        """
        Add gather to cache with LRU eviction (thread-safe).

        Args:
            gather_id: Gather ID
            gather_data: Gather seismic data
            gather_headers: Gather headers DataFrame
        """
        with self._cache_lock:
            # Evict least recently used if cache full
            while len(self._ensemble_cache) >= self._max_cached_ensembles:
                # Remove oldest (first) item
                oldest_id = next(iter(self._ensemble_cache))
                del self._ensemble_cache[oldest_id]

            # Add to cache (at end = most recently used)
            self._ensemble_cache[gather_id] = (gather_data, gather_headers)

    def _sort_gather_traces(self, gather_data: SeismicData,
                           gather_headers: pd.DataFrame) -> Tuple[SeismicData, pd.DataFrame]:
        """
        Sort traces within a gather by specified header keys.

        Args:
            gather_data: Gather seismic data
            gather_headers: Gather headers DataFrame

        Returns:
            Tuple of sorted gather_data and gather_headers
        """
        if not self.sort_keys:
            return gather_data, gather_headers

        # Validate sort keys exist in headers
        valid_sort_keys = [key for key in self.sort_keys if key in gather_headers.columns]

        if not valid_sort_keys:
            print(f"Warning: None of the sort keys {self.sort_keys} found in headers")
            return gather_data, gather_headers

        try:
            # Sort headers and get sort indices
            sorted_headers = gather_headers.sort_values(by=valid_sort_keys)
            sort_indices = sorted_headers.index.values

            # Sort traces using the same indices
            sorted_traces = gather_data.traces[:, sort_indices]

            # Create sorted SeismicData
            sorted_gather_data = SeismicData(
                traces=sorted_traces,
                sample_rate=gather_data.sample_rate,
                metadata={
                    **gather_data.metadata,
                    'sorted': True,
                    'sort_keys': valid_sort_keys,
                    'sort_indices': sort_indices.tolist()
                }
            )

            # Reset index for sorted headers
            sorted_headers = sorted_headers.reset_index(drop=True)

            return sorted_gather_data, sorted_headers

        except Exception as e:
            print(f"Warning: Failed to sort gather: {e}")
            return gather_data, gather_headers

    def _get_gather_info(self, gather_id: int) -> Dict:
        """
        Get information about a specific gather.

        Args:
            gather_id: Gather ID

        Returns:
            Dictionary with gather information
        """
        if not self.has_gathers():
            # Get n_traces from appropriate source
            if self.lazy_data is not None:
                n_traces = self.lazy_data.n_traces
            elif self.full_data is not None:
                n_traces = self.full_data.n_traces
            else:
                n_traces = 0

            return {
                'gather_id': 0,
                'n_traces': n_traces,
                'description': 'Full dataset'
            }

        ensemble = self.ensembles_df.iloc[gather_id]
        start_trace = int(ensemble['start_trace'])
        end_trace = int(ensemble['end_trace'])

        # Get ensemble key values
        key_values = {}
        if self.lazy_data is not None:
            # Lazy mode - load only first trace header
            try:
                first_header_df = self.lazy_data.get_headers([start_trace])
                first_header = first_header_df.iloc[0]
                for key in self.ensemble_keys:
                    if key in first_header:
                        key_values[key] = first_header[key]
            except Exception as e:
                print(f"Warning: Failed to load header for gather {gather_id}: {e}")
        elif self.headers_df is not None:
            # Full data mode
            first_header = self.headers_df.iloc[start_trace]
            for key in self.ensemble_keys:
                if key in first_header:
                    key_values[key] = first_header[key]

        # Create description
        if key_values:
            desc_parts = [f"{k}={v}" for k, v in key_values.items()]
            description = ", ".join(desc_parts)
        else:
            description = f"Gather {gather_id + 1}"

        return {
            'gather_id': gather_id,
            'ensemble_id': int(ensemble['ensemble_id']),
            'start_trace': start_trace,
            'end_trace': end_trace,
            'n_traces': int(ensemble['n_traces']),
            'key_values': key_values,
            'description': description
        }

    def next_gather(self) -> bool:
        """
        Navigate to next gather.

        Returns:
            True if navigation succeeded, False if already at last gather
        """
        if self.current_gather_id < self.n_gathers - 1:
            self.current_gather_id += 1
            self._emit_state_changes()
            self._trigger_prefetch(self.current_gather_id)  # Trigger background prefetch
            return True
        return False

    def previous_gather(self) -> bool:
        """
        Navigate to previous gather.

        Returns:
            True if navigation succeeded, False if already at first gather
        """
        if self.current_gather_id > 0:
            self.current_gather_id -= 1
            self._emit_state_changes()
            self._trigger_prefetch(self.current_gather_id)  # Trigger background prefetch
            return True
        return False

    def goto_gather(self, gather_id: int) -> bool:
        """
        Navigate to specific gather.

        Args:
            gather_id: Target gather ID (0-based)

        Returns:
            True if navigation succeeded, False if gather_id invalid
        """
        if 0 <= gather_id < self.n_gathers:
            self.current_gather_id = gather_id
            self._emit_state_changes()
            self._trigger_prefetch(self.current_gather_id)  # Trigger background prefetch
            return True
        return False

    def can_go_previous(self) -> bool:
        """Check if can navigate to previous gather."""
        return self.current_gather_id > 0

    def can_go_next(self) -> bool:
        """Check if can navigate to next gather."""
        return self.current_gather_id < self.n_gathers - 1

    def prefetch_adjacent(self):
        """
        Prefetch adjacent gathers (previous and next) into cache.

        This improves navigation performance by loading nearby gathers
        in the background before they are requested.

        Only works in lazy loading mode.
        """
        if self.lazy_data is None:
            return  # Only for lazy mode

        # Prefetch previous gather if available and not cached
        if self.can_go_previous():
            prev_id = self.current_gather_id - 1
            if prev_id not in self._ensemble_cache:
                try:
                    self._get_lazy_gather(prev_id)
                except Exception as e:
                    print(f"Warning: Failed to prefetch gather {prev_id}: {e}")

        # Prefetch next gather if available and not cached
        if self.can_go_next():
            next_id = self.current_gather_id + 1
            if next_id not in self._ensemble_cache:
                try:
                    self._get_lazy_gather(next_id)
                except Exception as e:
                    print(f"Warning: Failed to prefetch gather {next_id}: {e}")

    def get_cache_stats(self) -> Optional[Dict]:
        """
        Get ensemble cache statistics.

        Returns:
            Dictionary with cache stats or None if not using lazy mode
        """
        if self.lazy_data is None:
            return None

        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0.0

        return {
            'cache_size': len(self._ensemble_cache),
            'max_cache_size': self._max_cached_ensembles,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'total_requests': total_requests,
            'hit_rate': hit_rate,
            'cached_gather_ids': list(self._ensemble_cache.keys())
        }

    def _emit_state_changes(self):
        """Emit signals for state changes."""
        # Emit gather info
        gather_info = self._get_gather_info(self.current_gather_id)
        self.gather_changed.emit(self.current_gather_id, gather_info)

        # Emit navigation state
        self.navigation_state_changed.emit(
            self.can_go_previous(),
            self.can_go_next()
        )

    def get_statistics(self) -> Dict:
        """Get statistics about all gathers."""
        # Get total traces from appropriate source
        if self.lazy_data is not None:
            total_traces = self.lazy_data.n_traces
        elif self.full_data is not None:
            total_traces = self.full_data.n_traces
        else:
            total_traces = 0

        if not self.has_gathers():
            return {
                'n_gathers': 1,
                'total_traces': total_traces,
                'mode': 'single'
            }

        stats = {
            'n_gathers': self.n_gathers,
            'total_traces': total_traces,
            'ensemble_keys': self.ensemble_keys,
            'mode': 'multi-gather',
            'traces_per_gather': {
                'min': int(self.ensembles_df['n_traces'].min()),
                'max': int(self.ensembles_df['n_traces'].max()),
                'mean': float(self.ensembles_df['n_traces'].mean()),
            }
        }

        # Add cache stats if using lazy mode
        cache_stats = self.get_cache_stats()
        if cache_stats is not None:
            stats['cache'] = cache_stats

        return stats

    def _start_prefetch_thread(self):
        """Start background prefetch thread for lazy loading mode."""
        if self.lazy_data is None:
            return  # Only for lazy mode

        # Stop existing thread if any
        self._stop_prefetch_thread()

        # Create and start new thread
        self._stop_prefetch.clear()
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            daemon=True,
            name="GatherNavigator-Prefetch"
        )
        self._prefetch_thread.start()

    def _stop_prefetch_thread(self):
        """Stop background prefetch thread gracefully."""
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            self._stop_prefetch.set()
            self._prefetch_event.set()  # Wake thread to notice stop signal
            self._prefetch_thread.join(timeout=1.0)

    def _prefetch_worker(self):
        """
        Background worker thread that prefetches adjacent gathers.

        Runs continuously, waiting for prefetch signals from navigation.
        Prefetches previous 2 and next 2 gathers relative to current.
        """
        while not self._stop_prefetch.is_set():
            # Wait for prefetch signal or timeout
            self._prefetch_event.wait(timeout=0.5)
            self._prefetch_event.clear()

            if self._stop_prefetch.is_set():
                break

            # Get target gather ID
            if self._prefetch_gather_id is None:
                continue

            target_id = self._prefetch_gather_id

            # Prefetch adjacent gathers (prev 2 and next 2)
            prefetch_ids = []
            for offset in [-2, -1, 1, 2]:
                adj_id = target_id + offset
                if 0 <= adj_id < self.n_gathers:
                    # Check if not already cached
                    with self._cache_lock:
                        if adj_id not in self._ensemble_cache:
                            prefetch_ids.append(adj_id)

            # Load gathers that aren't cached
            for gather_id in prefetch_ids:
                if self._stop_prefetch.is_set():
                    break

                try:
                    # Load gather into cache
                    self._get_lazy_gather(gather_id)
                except Exception as e:
                    print(f"Warning: Prefetch failed for gather {gather_id}: {e}")

    def _trigger_prefetch(self, gather_id: int):
        """
        Trigger background prefetching for given gather.

        Args:
            gather_id: Gather ID around which to prefetch
        """
        if self.lazy_data is None or self._prefetch_thread is None:
            return

        self._prefetch_gather_id = gather_id
        self._prefetch_event.set()  # Wake prefetch thread

    def __del__(self):
        """Cleanup: stop prefetch thread gracefully."""
        try:
            self._stop_prefetch_thread()
        except Exception:
            pass  # Ignore errors during cleanup

    def __repr__(self) -> str:
        if self.has_gathers():
            return (f"GatherNavigator(gathers={self.n_gathers}, "
                   f"current={self.current_gather_id + 1}, "
                   f"keys={self.ensemble_keys})")
        else:
            return "GatherNavigator(single gather mode)"
