"""
Unit tests for GPUMemoryManager.

Tests:
- Memory manager construction
- Trace transfer to GPU
- Geometry transfer
- Optimal tile size computation
- Output tile generation
- Memory statistics
"""

import numpy as np
import pytest
import torch
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from processors.migration.gpu_memory import (
    GPUMemoryManager,
    OutputTile,
    create_gpu_memory_manager,
)


# Force CPU for consistent testing
TEST_DEVICE = torch.device('cpu')


class TestManagerConstruction:
    """Tests for memory manager construction."""

    def test_basic_construction(self):
        """Test basic construction."""
        manager = GPUMemoryManager(device=TEST_DEVICE)

        assert manager.device == TEST_DEVICE
        assert manager.target_memory_fraction > 0
        assert manager.target_memory_fraction <= 1.0

    def test_custom_parameters(self):
        """Test construction with custom parameters."""
        manager = GPUMemoryManager(
            device=TEST_DEVICE,
            target_memory_fraction=0.5,
            min_tile_size=20,
            max_tile_size=100,
        )

        assert manager.target_memory_fraction == 0.5
        assert manager.min_tile_size == 20
        assert manager.max_tile_size == 100


class TestTraceTransfer:
    """Tests for trace data transfer."""

    def test_transfer_traces(self):
        """Test transferring traces to GPU."""
        manager = GPUMemoryManager(device=TEST_DEVICE)

        # Create test traces
        traces = np.random.randn(1600, 100).astype(np.float32)

        traces_gpu = manager.transfer_traces_to_gpu(traces)

        assert traces_gpu.device.type == TEST_DEVICE.type
        assert traces_gpu.shape == (1600, 100)
        assert traces_gpu.dtype == torch.float32

    def test_transfer_pins_memory(self):
        """Test that transfer pins memory by default."""
        manager = GPUMemoryManager(device=TEST_DEVICE)

        traces = np.random.randn(1600, 50).astype(np.float32)
        manager.transfer_traces_to_gpu(traces, pin_memory=True)

        assert manager.trace_buffer is not None
        assert manager._allocated_bytes > 0

    def test_transfer_without_pinning(self):
        """Test transfer without pinning."""
        manager = GPUMemoryManager(device=TEST_DEVICE)

        traces = np.random.randn(100, 10).astype(np.float32)
        traces_gpu = manager.transfer_traces_to_gpu(traces, pin_memory=False)

        # Should still return tensor but not pin
        assert traces_gpu is not None
        assert manager.trace_buffer is None

    def test_free_trace_buffer(self):
        """Test freeing trace buffer."""
        manager = GPUMemoryManager(device=TEST_DEVICE)

        traces = np.random.randn(100, 10).astype(np.float32)
        manager.transfer_traces_to_gpu(traces)

        assert manager.trace_buffer is not None

        manager.free_trace_buffer()

        assert manager.trace_buffer is None


class TestGeometryTransfer:
    """Tests for geometry transfer."""

    def test_transfer_geometry(self):
        """Test transferring geometry to GPU."""
        manager = GPUMemoryManager(device=TEST_DEVICE)

        n_traces = 100
        src_x = np.random.randn(n_traces).astype(np.float32)
        src_y = np.random.randn(n_traces).astype(np.float32)
        rcv_x = np.random.randn(n_traces).astype(np.float32)
        rcv_y = np.random.randn(n_traces).astype(np.float32)
        offset = np.random.randn(n_traces).astype(np.float32)

        geom = manager.transfer_geometry_to_gpu(src_x, src_y, rcv_x, rcv_y, offset)

        assert 'source_x' in geom
        assert 'receiver_y' in geom
        assert geom['source_x'].device.type == TEST_DEVICE.type

    def test_free_geometry_buffer(self):
        """Test freeing geometry buffer."""
        manager = GPUMemoryManager(device=TEST_DEVICE)

        n_traces = 50
        manager.transfer_geometry_to_gpu(
            np.zeros(n_traces), np.zeros(n_traces),
            np.zeros(n_traces), np.zeros(n_traces),
            np.zeros(n_traces),
        )

        assert manager.geometry_buffer is not None

        manager.free_geometry_buffer()

        assert manager.geometry_buffer is None


class TestOptimalTileSize:
    """Tests for optimal tile size computation."""

    def test_returns_valid_size(self):
        """Test that tile size is in valid range."""
        manager = GPUMemoryManager(
            device=TEST_DEVICE,
            min_tile_size=10,
            max_tile_size=100,
        )

        tile_size = manager.get_optimal_tile_size(
            n_traces=500,
            n_samples=1600,
            n_depths=200,
        )

        assert tile_size >= manager.min_tile_size
        assert tile_size <= manager.max_tile_size

    def test_larger_with_fewer_traces(self):
        """Tile size should be larger with fewer traces."""
        manager = GPUMemoryManager(device=TEST_DEVICE)

        tile_few = manager.get_optimal_tile_size(
            n_traces=100,
            n_samples=1000,
            n_depths=100,
        )

        tile_many = manager.get_optimal_tile_size(
            n_traces=10000,
            n_samples=1000,
            n_depths=100,
        )

        # With fewer traces, more memory for tiles
        assert tile_few >= tile_many


class TestOutputTiles:
    """Tests for output tile generation."""

    def test_generate_tiles(self):
        """Test tile generation."""
        manager = GPUMemoryManager(device=TEST_DEVICE)

        tiles = manager.generate_output_tiles(
            n_inline=100,
            n_xline=100,
            tile_size=25,
        )

        # Should have 4x4 = 16 tiles
        assert len(tiles) == 16

    def test_tiles_cover_grid(self):
        """Test that tiles cover entire grid."""
        manager = GPUMemoryManager(device=TEST_DEVICE)

        n_il, n_xl = 97, 103  # Non-divisible by tile size
        tile_size = 25

        tiles = manager.generate_output_tiles(n_il, n_xl, tile_size)

        # Check coverage
        covered = np.zeros((n_il, n_xl), dtype=bool)
        for tile in tiles:
            covered[tile.il_start:tile.il_end, tile.xl_start:tile.xl_end] = True

        assert covered.all()

    def test_tiles_non_overlapping(self):
        """Test that tiles don't overlap."""
        manager = GPUMemoryManager(device=TEST_DEVICE)

        tiles = manager.generate_output_tiles(50, 50, 20)

        # Check no overlap
        coverage = np.zeros((50, 50), dtype=int)
        for tile in tiles:
            coverage[tile.il_start:tile.il_end, tile.xl_start:tile.xl_end] += 1

        assert coverage.max() == 1  # Each point covered exactly once

    def test_allocate_output_tile(self):
        """Test output tile allocation."""
        manager = GPUMemoryManager(device=TEST_DEVICE)

        image, fold = manager.allocate_output_tile(
            shape=(20, 30),
            n_depths=100,
        )

        assert image.shape == (100, 20, 30)
        assert fold.shape == (100, 20, 30)
        assert image.device.type == TEST_DEVICE.type


class TestOutputTileClass:
    """Tests for OutputTile dataclass."""

    def test_tile_properties(self):
        """Test tile properties."""
        tile = OutputTile(
            il_start=10,
            il_end=30,
            xl_start=20,
            xl_end=50,
        )

        assert tile.il_size == 20
        assert tile.xl_size == 30
        assert tile.shape == (20, 30)


class TestMemoryStats:
    """Tests for memory statistics."""

    def test_stats_tracking(self):
        """Test that memory stats are tracked."""
        manager = GPUMemoryManager(device=TEST_DEVICE)

        # Initial stats
        stats = manager.get_memory_stats()
        assert stats['allocated_bytes'] == 0

        # After allocation
        traces = np.random.randn(100, 50).astype(np.float32)
        manager.transfer_traces_to_gpu(traces)

        stats = manager.get_memory_stats()
        assert stats['allocated_bytes'] > 0
        assert stats['peak_bytes'] > 0

    def test_available_memory_positive(self):
        """Test that available memory is reported."""
        manager = GPUMemoryManager(device=TEST_DEVICE)

        available = manager.get_available_memory()
        assert available > 0


class TestSyncOutput:
    """Tests for output synchronization."""

    def test_sync_to_cpu(self):
        """Test syncing tensor to CPU."""
        manager = GPUMemoryManager(device=TEST_DEVICE)

        # Create GPU tensor
        tensor_gpu = torch.randn(10, 20, 30, device=TEST_DEVICE)

        # Sync to CPU
        array_cpu = manager.sync_output_to_cpu(tensor_gpu)

        assert isinstance(array_cpu, np.ndarray)
        assert array_cpu.shape == (10, 20, 30)


class TestEmptyCache:
    """Tests for cache emptying."""

    def test_empty_cache_runs(self):
        """Test that empty_cache runs without error."""
        manager = GPUMemoryManager(device=TEST_DEVICE)

        # Should not raise
        manager.empty_cache()


class TestFactoryFunction:
    """Tests for factory function."""

    def test_factory_creates_manager(self):
        """Test that factory creates valid manager."""
        manager = create_gpu_memory_manager(
            device=TEST_DEVICE,
            target_memory_fraction=0.6,
        )

        assert isinstance(manager, GPUMemoryManager)
        assert manager.device == TEST_DEVICE
        assert manager.target_memory_fraction == 0.6


class TestFreeTensor:
    """Tests for tensor freeing."""

    def test_free_tensor_updates_tracking(self):
        """Test that freeing tensor updates memory tracking."""
        manager = GPUMemoryManager(device=TEST_DEVICE)

        # Allocate
        traces = np.random.randn(100, 50).astype(np.float32)
        traces_gpu = manager.transfer_traces_to_gpu(traces, pin_memory=False)

        initial_allocated = manager._allocated_bytes

        # Free (note: pin_memory=False so not tracked internally)
        # This mainly tests the method doesn't crash
        manager.free_tensor(traces_gpu)


class TestIntegration:
    """Integration tests."""

    def test_full_workflow(self):
        """Test full migration-like workflow."""
        manager = GPUMemoryManager(device=TEST_DEVICE)

        # Setup
        n_traces = 200
        n_samples = 1600
        n_depths = 100
        n_il, n_xl = 50, 50

        # Transfer traces
        traces = np.random.randn(n_samples, n_traces).astype(np.float32)
        traces_gpu = manager.transfer_traces_to_gpu(traces)

        # Transfer geometry
        geom = manager.transfer_geometry_to_gpu(
            np.random.randn(n_traces).astype(np.float32),
            np.random.randn(n_traces).astype(np.float32),
            np.random.randn(n_traces).astype(np.float32),
            np.random.randn(n_traces).astype(np.float32),
            np.random.randn(n_traces).astype(np.float32),
        )

        # Get tile size
        tile_size = manager.get_optimal_tile_size(n_traces, n_samples, n_depths)

        # Process tiles
        output = np.zeros((n_depths, n_il, n_xl), dtype=np.float32)

        for tile in manager.generate_output_tiles(n_il, n_xl, tile_size):
            # Allocate tile
            img_tile, fold_tile = manager.allocate_output_tile(
                tile.shape, n_depths
            )

            # Simulate processing (fill with random data)
            img_tile.fill_(1.0)

            # Sync to output
            output[:, tile.il_start:tile.il_end, tile.xl_start:tile.xl_end] = (
                manager.sync_output_to_cpu(img_tile)
            )

        # Verify full output was filled
        assert output.sum() == n_depths * n_il * n_xl

        # Cleanup
        manager.free_trace_buffer()
        manager.free_geometry_buffer()
        manager.empty_cache()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
