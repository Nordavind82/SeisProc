"""
Window Cache with LRU (Least Recently Used) eviction policy.

Provides fast caching of data windows with automatic eviction based on:
- Maximum number of cached windows
- Maximum total memory usage
- LRU policy (least recently accessed windows evicted first)
"""
import numpy as np
from collections import OrderedDict
from typing import Optional, Tuple, Dict, Any
import threading


class WindowCache:
    """
    Thread-safe LRU cache for data windows.

    Features:
    - LRU eviction when count or memory limit exceeded
    - O(1) get/put operations using OrderedDict
    - Thread-safe with locks
    - Memory tracking using nbytes
    - Cache statistics (hits, misses, evictions)

    Example:
        >>> cache = WindowCache(max_windows=5, max_memory_mb=500)
        >>> cache.put((0, 1000, 0, 100), data_array)
        >>> data = cache.get((0, 1000, 0, 100))
        >>> print(cache.get_stats())
    """

    def __init__(self, max_windows: int = 5, max_memory_mb: float = 500.0):
        """
        Initialize window cache.

        Args:
            max_windows: Maximum number of windows to cache
            max_memory_mb: Maximum total memory in megabytes
        """
        self.max_windows = max_windows
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)

        # OrderedDict maintains insertion order and allows O(1) move_to_end
        self._cache: OrderedDict[Tuple, np.ndarray] = OrderedDict()
        self._sizes: Dict[Tuple, int] = {}  # Track memory size of each window

        # Current memory usage
        self._current_memory = 0

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        # Thread safety
        self._lock = threading.RLock()

    def get(self, key: Tuple) -> Optional[np.ndarray]:
        """
        Get cached window data.

        Args:
            key: Window identifier tuple (time_start, time_end, trace_start, trace_end)

        Returns:
            Cached numpy array if found, None otherwise
        """
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            else:
                self._misses += 1
                return None

    def put(self, key: Tuple, data: np.ndarray):
        """
        Add window data to cache.

        If cache is full (by count or memory), evicts least recently used windows.

        Args:
            key: Window identifier tuple (time_start, time_end, trace_start, trace_end)
            data: Numpy array to cache
        """
        with self._lock:
            data_size = data.nbytes

            # If this key already exists, remove old entry first
            if key in self._cache:
                self._current_memory -= self._sizes[key]
                del self._cache[key]
                del self._sizes[key]

            # Evict windows if necessary to make room
            while len(self._cache) >= self.max_windows or \
                  (self._current_memory + data_size > self.max_memory_bytes and len(self._cache) > 0):
                self._evict_lru()

            # Add new window
            self._cache[key] = data.copy()  # Store copy to avoid external modifications
            self._sizes[key] = data_size
            self._current_memory += data_size

    def _evict_lru(self):
        """Evict least recently used window (internal method)."""
        if not self._cache:
            return

        # OrderedDict: first item is least recently used
        lru_key = next(iter(self._cache))
        evicted_data = self._cache.pop(lru_key)
        evicted_size = self._sizes.pop(lru_key)

        self._current_memory -= evicted_size
        self._evictions += 1

    def clear(self):
        """Remove all cached windows."""
        with self._lock:
            self._cache.clear()
            self._sizes.clear()
            self._current_memory = 0

    def get_memory_usage(self) -> int:
        """
        Get current memory usage in bytes.

        Returns:
            Total memory used by cached windows
        """
        with self._lock:
            return self._current_memory

    def get_memory_usage_mb(self) -> float:
        """
        Get current memory usage in megabytes.

        Returns:
            Total memory used in MB
        """
        return self.get_memory_usage() / (1024 * 1024)

    def get_window_count(self) -> int:
        """
        Get number of cached windows.

        Returns:
            Count of windows currently in cache
        """
        with self._lock:
            return len(self._cache)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with hits, misses, evictions, hit_rate, memory_usage, window_count
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0

            return {
                'hits': self._hits,
                'misses': self._misses,
                'evictions': self._evictions,
                'hit_rate': hit_rate,
                'memory_usage_mb': self.get_memory_usage_mb(),
                'memory_limit_mb': self.max_memory_bytes / (1024 * 1024),
                'window_count': len(self._cache),
                'window_limit': self.max_windows,
                'total_requests': total_requests
            }

    def reset_stats(self):
        """Reset cache statistics (hits, misses, evictions)."""
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def contains(self, key: Tuple) -> bool:
        """
        Check if key is in cache without updating LRU order.

        Args:
            key: Window identifier tuple

        Returns:
            True if key exists in cache
        """
        with self._lock:
            return key in self._cache

    def __repr__(self) -> str:
        """String representation of cache state."""
        stats = self.get_stats()
        return (f"WindowCache(windows={stats['window_count']}/{stats['window_limit']}, "
                f"memory={stats['memory_usage_mb']:.1f}/{stats['memory_limit_mb']:.1f}MB, "
                f"hit_rate={stats['hit_rate']:.1f}%)")

    def __len__(self) -> int:
        """Return number of cached windows."""
        return self.get_window_count()
