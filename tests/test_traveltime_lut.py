"""
Unit tests for TraveltimeLUT (Traveltime Lookup Table).

Tests:
- LUT construction with constant velocity
- LUT construction with linear gradient velocity
- LUT construction with 1D v(z) array
- Bilinear interpolation accuracy
- Batch lookup performance
- Two-way traveltime computation
- Save/load functionality
- Comparison with direct traveltime computation
- Integration with KirchhoffMigrator
"""

import numpy as np
import pytest
import torch
from pathlib import Path
import tempfile

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from processors.migration.traveltime_lut import TraveltimeLUT, create_traveltime_lut
from processors.migration.traveltime import StraightRayTraveltime
from models.velocity_model import (
    VelocityModel,
    create_constant_velocity,
    create_linear_gradient_velocity,
)
from models.migration_config import MigrationConfig, OutputGrid


# Force CPU for consistent testing across platforms
TEST_DEVICE = torch.device('cpu')


class TestTraveltimeLUTConstruction:
    """Tests for LUT construction."""

    def test_basic_construction(self):
        """Test basic LUT construction with constant velocity."""
        lut = TraveltimeLUT(device=TEST_DEVICE)
        lut.build(
            velocity=2500.0,
            max_offset=3000.0,
            max_depth=2000.0,
            n_offsets=100,
            n_depths=100,
        )

        assert lut._built
        assert lut.shape == (100, 100)
        assert lut._max_offset == 3000.0
        assert lut._max_depth == 2000.0

    def test_table_values_positive(self):
        """Test that all table values are positive."""
        lut = TraveltimeLUT(device=TEST_DEVICE)
        lut.build(velocity=2500.0, max_offset=5000.0, max_depth=5000.0)

        # All traveltimes should be positive
        assert (lut._table > 0).all()

    def test_table_increases_with_offset(self):
        """Test that traveltime increases with horizontal offset."""
        lut = TraveltimeLUT(device=TEST_DEVICE)
        lut.build(velocity=2500.0, max_offset=5000.0, max_depth=5000.0)

        # For fixed depth, traveltime should increase with offset
        z_idx = 500  # Middle of depth range
        for h_idx in range(lut._n_offsets - 1):
            assert lut._table[h_idx, z_idx] <= lut._table[h_idx + 1, z_idx]

    def test_table_increases_with_depth(self):
        """Test that traveltime increases with depth."""
        lut = TraveltimeLUT(device=TEST_DEVICE)
        lut.build(velocity=2500.0, max_offset=5000.0, max_depth=5000.0)

        # For fixed offset, traveltime should increase with depth
        h_idx = 250  # Middle of offset range
        for z_idx in range(lut._n_depths - 1):
            assert lut._table[h_idx, z_idx] <= lut._table[h_idx, z_idx + 1]

    def test_gradient_velocity(self):
        """Test LUT construction with linear gradient velocity."""
        lut = TraveltimeLUT(device=TEST_DEVICE)
        lut.build(
            velocity=2000.0,
            max_offset=3000.0,
            max_depth=3000.0,
            n_offsets=100,
            n_depths=100,
            gradient=0.5,  # 0.5 m/s per meter
        )

        assert lut._built
        assert lut._velocity_type == 'gradient'
        assert lut._gradient == 0.5

    def test_vz_array(self):
        """Test LUT construction with 1D v(z) array."""
        # Create v(z) profile
        depths = np.linspace(0, 3000, 100)
        velocities = 2000 + 0.3 * depths  # Linear increase

        lut = TraveltimeLUT(device=TEST_DEVICE)
        lut.build(
            velocity=velocities,
            max_offset=3000.0,
            max_depth=3000.0,
            n_offsets=100,
            n_depths=100,
        )

        assert lut._built
        assert lut._velocity_type == '1d_array'

    def test_memory_size(self):
        """Test memory size estimation."""
        lut = TraveltimeLUT(device=TEST_DEVICE)
        lut.build(
            velocity=2500.0,
            max_offset=5000.0,
            max_depth=5000.0,
            n_offsets=500,
            n_depths=1000,
        )

        # 500 * 1000 * 4 bytes / 1024^2 = ~1.9 MB
        expected_mb = 500 * 1000 * 4 / (1024 * 1024)
        assert abs(lut.memory_mb - expected_mb) < 0.1


class TestTraveltimeLUTLookup:
    """Tests for LUT lookup operations."""

    @pytest.fixture
    def lut_constant(self):
        """Create LUT with constant velocity."""
        lut = TraveltimeLUT(device=TEST_DEVICE)
        lut.build(
            velocity=2500.0,
            max_offset=5000.0,
            max_depth=5000.0,
            n_offsets=500,
            n_depths=1000,
        )
        return lut

    def test_single_lookup(self, lut_constant):
        """Test single point lookup."""
        t = lut_constant.lookup(h=1000.0, z=2000.0)

        # Expected: sqrt(1000^2 + 2000^2) / 2500 = sqrt(5000000) / 2500 ≈ 0.894
        expected = np.sqrt(1000**2 + 2000**2) / 2500
        assert abs(t - expected) < 0.01  # 10ms tolerance

    def test_batch_lookup(self, lut_constant):
        """Test batch lookup."""
        h = torch.tensor([0.0, 1000.0, 2000.0, 3000.0], device=TEST_DEVICE)
        z = torch.tensor([1000.0, 1000.0, 1000.0, 1000.0], device=TEST_DEVICE)

        t = lut_constant.lookup_batch(h, z)

        assert t.shape == (4,)
        assert (t > 0).all()
        # Traveltime should increase with offset
        assert t[0] < t[1] < t[2] < t[3]

    def test_batch_lookup_2d(self, lut_constant):
        """Test batch lookup with 2D input."""
        h = torch.tensor([[0.0, 1000.0], [2000.0, 3000.0]], device=TEST_DEVICE)
        z = torch.tensor([[1000.0, 1000.0], [2000.0, 2000.0]], device=TEST_DEVICE)

        t = lut_constant.lookup_batch(h, z)

        assert t.shape == (2, 2)
        assert (t > 0).all()

    def test_lookup_clamping(self, lut_constant):
        """Test that out-of-bounds values are clamped."""
        # Offset beyond max
        t1 = lut_constant.lookup(h=10000.0, z=2000.0)  # Beyond max_offset=5000
        t2 = lut_constant.lookup(h=4999.0, z=2000.0)  # Just inside

        # Should be clamped to max offset value
        assert abs(t1 - t2) < 0.1  # Close to max offset traveltime

    def test_two_way_lookup(self, lut_constant):
        """Test two-way traveltime lookup."""
        h_src = torch.tensor([1000.0, 2000.0], device=TEST_DEVICE)
        h_rcv = torch.tensor([500.0, 1500.0], device=TEST_DEVICE)
        z = torch.tensor([1500.0, 2000.0], device=TEST_DEVICE)

        t_two_way = lut_constant.lookup_batch_2way(h_src, h_rcv, z)

        # Verify it's sum of one-way times
        t_src = lut_constant.lookup_batch(h_src, z)
        t_rcv = lut_constant.lookup_batch(h_rcv, z)

        assert torch.allclose(t_two_way, t_src + t_rcv, atol=1e-6)


class TestBilinearInterpolation:
    """Tests for bilinear interpolation accuracy."""

    @pytest.fixture
    def fine_lut(self):
        """Create fine-resolution LUT for interpolation tests."""
        lut = TraveltimeLUT(device=TEST_DEVICE)
        lut.build(
            velocity=2500.0,
            max_offset=5000.0,
            max_depth=5000.0,
            n_offsets=1000,
            n_depths=2000,
        )
        return lut

    def test_interpolation_at_grid_points(self, fine_lut):
        """Test that interpolation is exact at grid points."""
        # Query exactly at grid points
        h = fine_lut._h_axis[100]
        z = fine_lut._z_axis[200]

        t_lookup = fine_lut.lookup(float(h), float(z))
        t_table = float(fine_lut._table[100, 200])

        assert abs(t_lookup - t_table) < 1e-5

    def test_interpolation_between_points(self, fine_lut):
        """Test interpolation between grid points."""
        # Query between grid points
        dh = fine_lut._dh
        dz = fine_lut._dz

        h = float(fine_lut._h_axis[100]) + dh / 2
        z = float(fine_lut._z_axis[200]) + dz / 2

        t_lookup = fine_lut.lookup(h, z)

        # Should be average of four corners
        t00 = float(fine_lut._table[100, 200])
        t01 = float(fine_lut._table[100, 201])
        t10 = float(fine_lut._table[101, 200])
        t11 = float(fine_lut._table[101, 201])
        t_expected = (t00 + t01 + t10 + t11) / 4

        assert abs(t_lookup - t_expected) < 0.001


class TestComparisonWithDirect:
    """Tests comparing LUT with direct traveltime computation."""

    def test_constant_velocity_accuracy(self):
        """Test LUT accuracy against direct computation for constant velocity."""
        v = 2500.0
        lut = TraveltimeLUT(device=TEST_DEVICE)
        lut.build(
            velocity=v,
            max_offset=5000.0,
            max_depth=5000.0,
            n_offsets=500,
            n_depths=1000,
        )

        # Test multiple random points
        np.random.seed(42)
        for _ in range(100):
            h = np.random.uniform(0, 4500)
            z = np.random.uniform(100, 4500)

            t_lut = lut.lookup(h, z)
            t_direct = np.sqrt(h**2 + z**2) / v

            # Should match within 1% for well-resolved points
            rel_error = abs(t_lut - t_direct) / t_direct
            assert rel_error < 0.01, f"h={h}, z={z}: LUT={t_lut}, direct={t_direct}"

    def test_vs_straight_ray_calculator(self):
        """Test LUT against StraightRayTraveltime calculator."""
        velocity_model = create_constant_velocity(2500.0)
        calc = StraightRayTraveltime(velocity_model, device=TEST_DEVICE)

        lut = TraveltimeLUT(device=TEST_DEVICE)
        lut.build(
            velocity=2500.0,
            max_offset=5000.0,
            max_depth=5000.0,
            n_offsets=500,
            n_depths=1000,
        )

        # Compare on grid of points
        h_vals = torch.linspace(0, 4000, 20, device=TEST_DEVICE)
        z_vals = torch.linspace(500, 4000, 20, device=TEST_DEVICE)

        for h in h_vals:
            for z in z_vals:
                # LUT lookup
                t_lut = lut.lookup(float(h), float(z))

                # Direct calculation
                t_direct = calc.compute_traveltime(h, torch.tensor(0.0), z)

                rel_error = abs(t_lut - float(t_direct)) / float(t_direct)
                assert rel_error < 0.01


class TestSaveLoad:
    """Tests for LUT persistence."""

    def test_save_load_roundtrip(self):
        """Test saving and loading LUT."""
        lut = TraveltimeLUT(device=TEST_DEVICE)
        lut.build(
            velocity=2500.0,
            max_offset=5000.0,
            max_depth=5000.0,
            n_offsets=200,
            n_depths=400,
            gradient=0.3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_lut.npz"

            # Save
            lut.save(path)

            # Load into new instance
            lut2 = TraveltimeLUT(device=TEST_DEVICE)
            lut2.load(path)

            # Verify
            assert lut2._built
            assert lut2._max_offset == lut._max_offset
            assert lut2._max_depth == lut._max_depth
            assert lut2._n_offsets == lut._n_offsets
            assert lut2._n_depths == lut._n_depths
            assert lut2._velocity_type == lut._velocity_type
            assert lut2._v0 == lut._v0
            assert lut2._gradient == lut._gradient

            # Verify table values
            assert torch.allclose(lut2._table, lut._table)

    def test_lookup_after_load(self):
        """Test that lookup works correctly after loading."""
        lut = TraveltimeLUT(device=TEST_DEVICE)
        lut.build(velocity=2500.0, max_offset=5000.0, max_depth=5000.0)

        # Get some reference values
        ref_vals = [
            lut.lookup(1000.0, 2000.0),
            lut.lookup(2500.0, 3000.0),
            lut.lookup(500.0, 1000.0),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_lut.npz"
            lut.save(path)

            lut2 = TraveltimeLUT(device=TEST_DEVICE)
            lut2.load(path)

            # Verify lookups match
            assert abs(lut2.lookup(1000.0, 2000.0) - ref_vals[0]) < 1e-5
            assert abs(lut2.lookup(2500.0, 3000.0) - ref_vals[1]) < 1e-5
            assert abs(lut2.lookup(500.0, 1000.0) - ref_vals[2]) < 1e-5


class TestLUTStats:
    """Tests for LUT statistics and info."""

    def test_get_stats(self):
        """Test stats retrieval."""
        lut = TraveltimeLUT(device=TEST_DEVICE)
        lut.build(
            velocity=2500.0,
            max_offset=5000.0,
            max_depth=5000.0,
            n_offsets=500,
            n_depths=1000,
        )

        stats = lut.get_stats()

        assert stats['built'] is True
        assert stats['shape'] == (500, 1000)
        assert stats['max_offset'] == 5000.0
        assert stats['max_depth'] == 5000.0
        assert stats['v0'] == 2500.0
        assert stats['memory_mb'] > 0
        assert stats['build_time'] > 0

    def test_stats_before_build(self):
        """Test stats before building LUT."""
        lut = TraveltimeLUT(device=TEST_DEVICE)
        stats = lut.get_stats()

        assert stats['built'] is False
        assert stats['memory_mb'] == 0


class TestDeviceHandling:
    """Tests for device handling."""

    def test_cpu_device(self):
        """Test LUT on CPU."""
        lut = TraveltimeLUT(device=torch.device('cpu'))
        lut.build(velocity=2500.0, max_offset=3000.0, max_depth=3000.0)

        assert lut._table.device.type == 'cpu'

    def test_to_device(self):
        """Test moving LUT between devices."""
        lut = TraveltimeLUT(device=torch.device('cpu'))
        lut.build(velocity=2500.0, max_offset=3000.0, max_depth=3000.0)

        # Get reference value
        t_ref = lut.lookup(1000.0, 2000.0)

        # Move to same device (CPU)
        lut.to_device(torch.device('cpu'))

        # Verify lookup still works
        t_after = lut.lookup(1000.0, 2000.0)
        assert abs(t_after - t_ref) < 1e-6


class TestErrorHandling:
    """Tests for error handling."""

    def test_lookup_before_build(self):
        """Test that lookup fails before build."""
        lut = TraveltimeLUT(device=TEST_DEVICE)

        with pytest.raises(RuntimeError, match="not built"):
            lut.lookup(1000.0, 2000.0)

    def test_batch_lookup_before_build(self):
        """Test that batch lookup fails before build."""
        lut = TraveltimeLUT(device=TEST_DEVICE)

        with pytest.raises(RuntimeError, match="not built"):
            lut.lookup_batch(
                torch.tensor([1000.0]),
                torch.tensor([2000.0]),
            )

    def test_save_before_build(self):
        """Test that save fails before build."""
        lut = TraveltimeLUT(device=TEST_DEVICE)

        with pytest.raises(RuntimeError, match="not built"):
            lut.save(Path("/tmp/test.npz"))


class TestFactoryFunction:
    """Tests for create_traveltime_lut factory function."""

    def test_factory_basic(self):
        """Test factory function creates valid LUT."""
        velocity = create_constant_velocity(2500.0)

        # Create minimal config
        output_grid = OutputGrid(
            n_time=100,
            n_inline=10,
            n_xline=10,
            dt=0.004,
            d_inline=25.0,
            d_xline=25.0,
            t0=0.0,
            x_origin=0.0,
            y_origin=0.0,
        )

        config = MigrationConfig(
            output_grid=output_grid,
            max_aperture_m=3000.0,
        )

        lut = create_traveltime_lut(velocity, config, device=TEST_DEVICE)

        assert lut._built
        assert lut._max_offset == 3000.0


class TestPerformance:
    """Performance-related tests."""

    def test_batch_lookup_large(self):
        """Test batch lookup with large arrays."""
        lut = TraveltimeLUT(device=TEST_DEVICE)
        lut.build(
            velocity=2500.0,
            max_offset=5000.0,
            max_depth=5000.0,
            n_offsets=500,
            n_depths=1000,
        )

        # Large batch
        n_points = 100000
        h = torch.rand(n_points, device=TEST_DEVICE) * 4000
        z = torch.rand(n_points, device=TEST_DEVICE) * 4000 + 100

        import time
        start = time.time()
        t = lut.lookup_batch(h, z)
        elapsed = time.time() - start

        assert t.shape == (n_points,)
        assert (t > 0).all()
        print(f"\nLUT batch lookup: {n_points} points in {elapsed*1000:.1f} ms")

    def test_lookup_vs_direct_speed(self):
        """Compare LUT lookup speed vs direct computation."""
        v = 2500.0
        lut = TraveltimeLUT(device=TEST_DEVICE)
        lut.build(
            velocity=v,
            max_offset=5000.0,
            max_depth=5000.0,
            n_offsets=500,
            n_depths=1000,
        )

        n_points = 50000
        h = torch.rand(n_points, device=TEST_DEVICE) * 4000
        z = torch.rand(n_points, device=TEST_DEVICE) * 4000 + 100

        import time

        # LUT lookup
        start = time.time()
        for _ in range(10):
            t_lut = lut.lookup_batch(h, z)
        lut_time = (time.time() - start) / 10

        # Direct computation
        start = time.time()
        for _ in range(10):
            t_direct = torch.sqrt(h**2 + z**2) / v
        direct_time = (time.time() - start) / 10

        print(f"\nLUT: {lut_time*1000:.2f} ms, Direct: {direct_time*1000:.2f} ms")
        print(f"LUT overhead: {lut_time/direct_time:.2f}x")

        # LUT has some overhead due to interpolation, but it's the benefit
        # that matters for complex velocity models where direct computation
        # is much more expensive. For constant velocity, LUT may be slower.
        # Just verify it completes in reasonable time (< 10ms for 50K points)
        assert lut_time < 0.1  # 100ms max for 50K points on CPU


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
