"""
Tests for Time-Domain Migration and KD-tree Spatial Index.

Tests:
- TraceSpatialIndex class (KD-tree)
- migrate_kirchhoff_time_domain kernel
- MigrationEngine.migrate_bin_time_domain method
- ConfigAdapter time-domain params
"""

import pytest
import numpy as np
import torch
import logging

from processors.migration.kirchhoff_kernel import (
    migrate_kirchhoff_time_domain,
    TraceSpatialIndex,
    SCIPY_AVAILABLE,
)
from processors.migration.migration_engine import MigrationEngine
from processors.migration.config_adapter import ConfigAdapter, MigrationParams


# Set up logging for tests
logging.basicConfig(level=logging.INFO)


class TestTraceSpatialIndex:
    """Tests for KD-tree spatial index."""

    def test_build_index(self):
        """Test building the spatial index."""
        if not SCIPY_AVAILABLE:
            pytest.skip("scipy not available")

        index = TraceSpatialIndex()

        # Create simple geometry
        source_x = np.array([0, 100, 200, 300, 400], dtype=np.float32)
        source_y = np.array([50, 50, 50, 50, 50], dtype=np.float32)
        receiver_x = source_x.copy()
        receiver_y = source_y.copy()

        index.build(source_x, source_y, receiver_x, receiver_y)

        assert index.is_built
        assert index.n_traces == 5

    def test_query_aperture(self):
        """Test aperture query returns correct traces."""
        if not SCIPY_AVAILABLE:
            pytest.skip("scipy not available")

        index = TraceSpatialIndex()

        # Traces at x = 0, 100, 200, 300, 400
        source_x = np.array([0, 100, 200, 300, 400], dtype=np.float32)
        source_y = np.array([0, 0, 0, 0, 0], dtype=np.float32)
        receiver_x = source_x.copy()
        receiver_y = source_y.copy()

        index.build(source_x, source_y, receiver_x, receiver_y)

        # Query at x=200, y=0 with radius 150 should get traces 1, 2, 3
        result = index.query_aperture(200.0, 0.0, 150.0)

        assert len(result) == 3
        assert 1 in result
        assert 2 in result
        assert 3 in result

    def test_query_tile(self):
        """Test tile query returns traces for entire tile area."""
        if not SCIPY_AVAILABLE:
            pytest.skip("scipy not available")

        index = TraceSpatialIndex()

        # Create 10x10 grid of traces
        x = np.arange(0, 1000, 100, dtype=np.float32)
        y = np.arange(0, 1000, 100, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        source_x = xx.flatten()
        source_y = yy.flatten()

        index.build(source_x, source_y, source_x, source_y)

        # Query tile in center with 200m aperture
        tile_x = torch.tensor([400, 500, 600], dtype=torch.float32)
        tile_y = torch.tensor([400, 500, 600], dtype=torch.float32)

        result = index.query_tile(tile_x, tile_y, max_aperture=200.0)

        # Should return several traces
        assert len(result) > 0
        assert len(result) < 100  # Not all traces

    def test_fallback_when_not_built(self):
        """Test that unbuilt index returns all traces."""
        index = TraceSpatialIndex()
        index.n_traces = 50

        result = index.query_aperture(0.0, 0.0, 100.0)

        assert len(result) == 50


class TestTimeDomainMigration:
    """Tests for time-domain migration kernel."""

    @pytest.fixture
    def simple_data(self):
        """Create simple test data."""
        device = torch.device('cpu')
        n_samples = 100
        n_traces = 50
        n_il, n_xl = 10, 10

        # Simple traces with impulse
        traces = torch.zeros(n_samples, n_traces, device=device)
        traces[25, :] = 1.0  # Impulse at sample 25

        # Simple geometry - traces centered in grid
        source_x = torch.linspace(0, 250, n_traces, device=device)
        source_y = torch.full((n_traces,), 125.0, device=device)
        receiver_x = source_x.clone()
        receiver_y = source_y.clone()

        # Output grid
        output_x = torch.linspace(0, 250, n_il * n_xl, device=device)
        output_y = torch.linspace(0, 250, n_il * n_xl, device=device)

        # Time axis
        dt_ms = 4.0
        t_min_ms = 0.0
        time_axis_ms = torch.arange(n_samples, device=device) * dt_ms + t_min_ms

        return {
            'traces': traces,
            'source_x': source_x,
            'source_y': source_y,
            'receiver_x': receiver_x,
            'receiver_y': receiver_y,
            'output_x': output_x,
            'output_y': output_y,
            'time_axis_ms': time_axis_ms,
            'dt_ms': dt_ms,
            't_min_ms': t_min_ms,
            'n_il': n_il,
            'n_xl': n_xl,
        }

    def test_returns_correct_shapes(self, simple_data):
        """Test that output shapes are correct."""
        image, fold = migrate_kirchhoff_time_domain(
            traces=simple_data['traces'],
            source_x=simple_data['source_x'],
            source_y=simple_data['source_y'],
            receiver_x=simple_data['receiver_x'],
            receiver_y=simple_data['receiver_y'],
            output_x=simple_data['output_x'],
            output_y=simple_data['output_y'],
            time_axis_ms=simple_data['time_axis_ms'],
            velocity=2500.0,
            dt_ms=simple_data['dt_ms'],
            t_min_ms=simple_data['t_min_ms'],
            max_aperture_m=500.0,
            max_angle_deg=60.0,
            n_il=simple_data['n_il'],
            n_xl=simple_data['n_xl'],
            tile_size=25,
        )

        n_times = len(simple_data['time_axis_ms'])
        n_il = simple_data['n_il']
        n_xl = simple_data['n_xl']

        assert image.shape == (n_times, n_il, n_xl)
        assert fold.shape == (n_times, n_il, n_xl)

    def test_produces_nonzero_output(self, simple_data):
        """Test that migration produces non-zero output."""
        image, fold = migrate_kirchhoff_time_domain(
            traces=simple_data['traces'],
            source_x=simple_data['source_x'],
            source_y=simple_data['source_y'],
            receiver_x=simple_data['receiver_x'],
            receiver_y=simple_data['receiver_y'],
            output_x=simple_data['output_x'],
            output_y=simple_data['output_y'],
            time_axis_ms=simple_data['time_axis_ms'],
            velocity=2500.0,
            dt_ms=simple_data['dt_ms'],
            t_min_ms=simple_data['t_min_ms'],
            max_aperture_m=500.0,
            max_angle_deg=60.0,
            n_il=simple_data['n_il'],
            n_xl=simple_data['n_xl'],
            tile_size=25,
        )

        assert image.abs().max() > 0
        assert fold.max() > 0

    def test_time_dependent_aperture_option(self, simple_data):
        """Test that time-dependent aperture option works."""
        # With time-dependent aperture
        image_td, fold_td = migrate_kirchhoff_time_domain(
            traces=simple_data['traces'],
            source_x=simple_data['source_x'],
            source_y=simple_data['source_y'],
            receiver_x=simple_data['receiver_x'],
            receiver_y=simple_data['receiver_y'],
            output_x=simple_data['output_x'],
            output_y=simple_data['output_y'],
            time_axis_ms=simple_data['time_axis_ms'],
            velocity=2500.0,
            dt_ms=simple_data['dt_ms'],
            t_min_ms=simple_data['t_min_ms'],
            max_aperture_m=500.0,
            max_angle_deg=60.0,
            n_il=simple_data['n_il'],
            n_xl=simple_data['n_xl'],
            tile_size=25,
            use_time_dependent_aperture=True,
        )

        # Without time-dependent aperture
        image_const, fold_const = migrate_kirchhoff_time_domain(
            traces=simple_data['traces'],
            source_x=simple_data['source_x'],
            source_y=simple_data['source_y'],
            receiver_x=simple_data['receiver_x'],
            receiver_y=simple_data['receiver_y'],
            output_x=simple_data['output_x'],
            output_y=simple_data['output_y'],
            time_axis_ms=simple_data['time_axis_ms'],
            velocity=2500.0,
            dt_ms=simple_data['dt_ms'],
            t_min_ms=simple_data['t_min_ms'],
            max_aperture_m=500.0,
            max_angle_deg=60.0,
            n_il=simple_data['n_il'],
            n_xl=simple_data['n_xl'],
            tile_size=25,
            use_time_dependent_aperture=False,
        )

        # Both should produce valid output
        assert image_td.abs().max() > 0
        assert image_const.abs().max() > 0

    def test_profiling_runs(self, simple_data):
        """Test that profiling option works."""
        image, fold = migrate_kirchhoff_time_domain(
            traces=simple_data['traces'],
            source_x=simple_data['source_x'],
            source_y=simple_data['source_y'],
            receiver_x=simple_data['receiver_x'],
            receiver_y=simple_data['receiver_y'],
            output_x=simple_data['output_x'],
            output_y=simple_data['output_y'],
            time_axis_ms=simple_data['time_axis_ms'],
            velocity=2500.0,
            dt_ms=simple_data['dt_ms'],
            t_min_ms=simple_data['t_min_ms'],
            max_aperture_m=500.0,
            max_angle_deg=60.0,
            n_il=simple_data['n_il'],
            n_xl=simple_data['n_xl'],
            tile_size=25,
            enable_profiling=True,
        )

        # Should complete without error
        assert image.shape[0] > 0


class TestMigrationEngineTimeDomain:
    """Tests for MigrationEngine.migrate_bin_time_domain."""

    def test_engine_has_time_domain_method(self):
        """Test that engine has the time-domain method."""
        engine = MigrationEngine(device=torch.device('cpu'))
        assert hasattr(engine, 'migrate_bin_time_domain')

    def test_time_domain_migration_runs(self):
        """Test that time-domain migration runs through engine."""
        engine = MigrationEngine(device=torch.device('cpu'))

        # Create simple data
        n_samples, n_traces = 100, 50
        n_il, n_xl = 10, 10

        traces = np.random.randn(n_samples, n_traces).astype(np.float32)
        source_x = np.linspace(0, 250, n_traces).astype(np.float32)
        source_y = np.full(n_traces, 125.0, dtype=np.float32)
        receiver_x = source_x.copy()
        receiver_y = source_y.copy()

        image, fold = engine.migrate_bin_time_domain(
            traces=traces,
            source_x=source_x,
            source_y=source_y,
            receiver_x=receiver_x,
            receiver_y=receiver_y,
            origin_x=0.0,
            origin_y=0.0,
            il_spacing=25.0,
            xl_spacing=25.0,
            azimuth_deg=0.0,
            n_il=n_il,
            n_xl=n_xl,
            dt_ms=4.0,
            t_min_ms=0.0,
            n_times=n_samples,
            velocity_mps=2500.0,
            max_aperture_m=500.0,
            max_angle_deg=60.0,
            tile_size=25,
        )

        assert image.shape == (n_samples, n_il, n_xl)
        assert fold.shape == (n_samples, n_il, n_xl)


class TestConfigAdapterTimeDomain:
    """Tests for ConfigAdapter time-domain support."""

    def test_params_include_advanced_fields(self):
        """Test that MigrationParams includes advanced fields."""
        config = {
            'x_origin': 0.0,
            'y_origin': 0.0,
            'output_bin_il': 25.0,
            'output_bin_xl': 25.0,
            'inline_min': 1,
            'inline_max': 10,
            'xline_min': 1,
            'xline_max': 10,
            'dt_ms': 4.0,
            'time_min_ms': 0.0,
            'time_max_ms': 400.0,
            'velocity_v0': 2500.0,
            'max_aperture_m': 3000.0,
            'max_angle_deg': 60.0,
            'tile_size': 200,
            'use_time_domain': True,
            'use_kdtree': True,
            'sample_batch_size': 150,
        }

        adapter = ConfigAdapter(config)
        p = adapter.params

        assert p.tile_size == 200
        assert p.use_time_domain is True
        assert p.use_kdtree is True
        assert p.sample_batch_size == 150

    def test_get_time_domain_params(self):
        """Test get_time_domain_params returns correct dict."""
        config = {
            'x_origin': 100.0,
            'y_origin': 200.0,
            'output_bin_il': 25.0,
            'output_bin_xl': 25.0,
            'inline_min': 1,
            'inline_max': 10,
            'xline_min': 1,
            'xline_max': 10,
            'dt_ms': 4.0,
            'time_min_ms': 0.0,
            'time_max_ms': 400.0,
            'velocity_v0': 2500.0,
            'max_aperture_m': 3000.0,
            'max_angle_deg': 60.0,
            'tile_size': 100,
        }

        adapter = ConfigAdapter(config)

        traces = np.zeros((101, 50), dtype=np.float32)
        source_x = np.zeros(50, dtype=np.float32)
        source_y = np.zeros(50, dtype=np.float32)
        receiver_x = np.zeros(50, dtype=np.float32)
        receiver_y = np.zeros(50, dtype=np.float32)

        params = adapter.get_time_domain_params(
            traces=traces,
            source_x=source_x,
            source_y=source_y,
            receiver_x=receiver_x,
            receiver_y=receiver_y,
        )

        assert 'n_times' in params
        assert 'tile_size' in params
        assert params['n_times'] == 101
        assert params['origin_x'] == 100.0
        assert params['origin_y'] == 200.0

    def test_advanced_params_defaults(self):
        """Test that advanced parameters have sensible defaults."""
        config = {
            'x_origin': 0.0,
            'y_origin': 0.0,
            'output_bin_il': 25.0,
            'output_bin_xl': 25.0,
            'inline_min': 1,
            'inline_max': 10,
            'xline_min': 1,
            'xline_max': 10,
            'dt_ms': 4.0,
            'time_min_ms': 0.0,
            'time_max_ms': 400.0,
            'velocity_v0': 2500.0,
            'max_aperture_m': 3000.0,
            'max_angle_deg': 60.0,
            # No advanced params specified
        }

        adapter = ConfigAdapter(config)
        p = adapter.params

        assert p.tile_size == 100  # default
        assert p.use_time_domain is False  # default
        assert p.use_kdtree is False  # default
        assert p.sample_batch_size == 200  # default
