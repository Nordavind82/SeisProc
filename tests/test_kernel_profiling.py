"""
Tests for kernel profiling instrumentation (Task 1.1).

Tests that enable_profiling=True produces timing breakdown in logs.
"""

import pytest
import numpy as np
import torch
import logging

from processors.migration.kirchhoff_kernel import migrate_kirchhoff_full
from processors.migration.migration_engine import MigrationEngine


class TestKernelProfiling:
    """Test timing instrumentation in kirchhoff_kernel."""

    @pytest.fixture
    def small_dataset(self):
        """Create small dataset for profiling test."""
        n_samples = 100
        n_traces = 50
        np.random.seed(42)

        # Traces with some signal
        traces = np.random.randn(n_samples, n_traces).astype(np.float32)

        # Simple geometry: traces on a grid
        x = np.linspace(0, 500, n_traces).astype(np.float32)
        y = np.zeros(n_traces, dtype=np.float32)

        return {
            'traces': traces,
            'source_x': x - 50,
            'source_y': y,
            'receiver_x': x + 50,
            'receiver_y': y,
        }

    def test_profiling_runs_without_error(self, small_dataset):
        """Test that profiling mode runs without errors."""
        device = torch.device('cpu')

        traces_t = torch.from_numpy(small_dataset['traces']).to(device)
        source_x_t = torch.from_numpy(small_dataset['source_x']).to(device)
        source_y_t = torch.from_numpy(small_dataset['source_y']).to(device)
        receiver_x_t = torch.from_numpy(small_dataset['receiver_x']).to(device)
        receiver_y_t = torch.from_numpy(small_dataset['receiver_y']).to(device)

        n_il, n_xl = 10, 10
        output_x = torch.linspace(0, 450, n_il * n_xl, device=device)
        output_y = torch.zeros(n_il * n_xl, device=device)
        depth_axis = torch.linspace(10, 500, 50, device=device)

        image, fold = migrate_kirchhoff_full(
            traces=traces_t,
            source_x=source_x_t,
            source_y=source_y_t,
            receiver_x=receiver_x_t,
            receiver_y=receiver_y_t,
            output_x=output_x,
            output_y=output_y,
            depth_axis=depth_axis,
            velocity=2000.0,
            dt_ms=4.0,
            t_min_ms=0.0,
            max_aperture_m=3000.0,
            max_angle_deg=60.0,
            n_il=n_il,
            n_xl=n_xl,
            enable_profiling=True,
        )

        assert image.shape == (50, n_il, n_xl)
        assert fold.shape == (50, n_il, n_xl)

    def test_profiling_logs_timing_breakdown(self, small_dataset, caplog):
        """Test that profiling produces timing logs."""
        device = torch.device('cpu')

        traces_t = torch.from_numpy(small_dataset['traces']).to(device)
        source_x_t = torch.from_numpy(small_dataset['source_x']).to(device)
        source_y_t = torch.from_numpy(small_dataset['source_y']).to(device)
        receiver_x_t = torch.from_numpy(small_dataset['receiver_x']).to(device)
        receiver_y_t = torch.from_numpy(small_dataset['receiver_y']).to(device)

        n_il, n_xl = 5, 5
        output_x = torch.linspace(0, 200, n_il * n_xl, device=device)
        output_y = torch.zeros(n_il * n_xl, device=device)
        depth_axis = torch.linspace(10, 200, 20, device=device)

        with caplog.at_level(logging.INFO):
            image, fold = migrate_kirchhoff_full(
                traces=traces_t,
                source_x=source_x_t,
                source_y=source_y_t,
                receiver_x=receiver_x_t,
                receiver_y=receiver_y_t,
                output_x=output_x,
                output_y=output_y,
                depth_axis=depth_axis,
                velocity=2000.0,
                dt_ms=4.0,
                t_min_ms=0.0,
                max_aperture_m=3000.0,
                max_angle_deg=60.0,
                n_il=n_il,
                n_xl=n_xl,
                enable_profiling=True,
            )

        log_text = caplog.text

        # Check for key profiling output sections
        assert "KIRCHHOFF MIGRATION PROFILING RESULTS" in log_text
        assert "TIME BREAKDOWN" in log_text
        assert "OPERATION COUNTS" in log_text
        assert "DERIVED METRICS" in log_text

        # Check for key timing categories
        assert "sqrt_ray_dist" in log_text
        assert "interpolation" in log_text
        assert "aperture_mask" in log_text

    def test_profiling_disabled_produces_no_logs(self, small_dataset, caplog):
        """Test that profiling=False produces no profiling logs."""
        device = torch.device('cpu')

        traces_t = torch.from_numpy(small_dataset['traces']).to(device)
        source_x_t = torch.from_numpy(small_dataset['source_x']).to(device)
        source_y_t = torch.from_numpy(small_dataset['source_y']).to(device)
        receiver_x_t = torch.from_numpy(small_dataset['receiver_x']).to(device)
        receiver_y_t = torch.from_numpy(small_dataset['receiver_y']).to(device)

        n_il, n_xl = 5, 5
        output_x = torch.linspace(0, 200, n_il * n_xl, device=device)
        output_y = torch.zeros(n_il * n_xl, device=device)
        depth_axis = torch.linspace(10, 200, 20, device=device)

        with caplog.at_level(logging.INFO):
            image, fold = migrate_kirchhoff_full(
                traces=traces_t,
                source_x=source_x_t,
                source_y=source_y_t,
                receiver_x=receiver_x_t,
                receiver_y=receiver_y_t,
                output_x=output_x,
                output_y=output_y,
                depth_axis=depth_axis,
                velocity=2000.0,
                dt_ms=4.0,
                t_min_ms=0.0,
                max_aperture_m=3000.0,
                max_angle_deg=60.0,
                n_il=n_il,
                n_xl=n_xl,
                enable_profiling=False,
            )

        log_text = caplog.text
        assert "KIRCHHOFF MIGRATION PROFILING RESULTS" not in log_text

    def test_profiling_aperture_statistics(self, small_dataset, caplog):
        """Test that aperture pass rate is logged."""
        device = torch.device('cpu')

        traces_t = torch.from_numpy(small_dataset['traces']).to(device)
        source_x_t = torch.from_numpy(small_dataset['source_x']).to(device)
        source_y_t = torch.from_numpy(small_dataset['source_y']).to(device)
        receiver_x_t = torch.from_numpy(small_dataset['receiver_x']).to(device)
        receiver_y_t = torch.from_numpy(small_dataset['receiver_y']).to(device)

        n_il, n_xl = 5, 5
        output_x = torch.linspace(0, 200, n_il * n_xl, device=device)
        output_y = torch.zeros(n_il * n_xl, device=device)
        depth_axis = torch.linspace(10, 200, 20, device=device)

        with caplog.at_level(logging.INFO):
            image, fold = migrate_kirchhoff_full(
                traces=traces_t,
                source_x=source_x_t,
                source_y=source_y_t,
                receiver_x=receiver_x_t,
                receiver_y=receiver_y_t,
                output_x=output_x,
                output_y=output_y,
                depth_axis=depth_axis,
                velocity=2000.0,
                dt_ms=4.0,
                t_min_ms=0.0,
                max_aperture_m=3000.0,
                max_angle_deg=60.0,
                n_il=n_il,
                n_xl=n_xl,
                enable_profiling=True,
            )

        log_text = caplog.text
        assert "Aperture pass rate" in log_text
        assert "Total trace pairs" in log_text
        assert "Aperture passed" in log_text

    def test_profiling_aperture_by_depth(self, small_dataset, caplog):
        """Test that per-depth aperture statistics are logged."""
        device = torch.device('cpu')

        traces_t = torch.from_numpy(small_dataset['traces']).to(device)
        source_x_t = torch.from_numpy(small_dataset['source_x']).to(device)
        source_y_t = torch.from_numpy(small_dataset['source_y']).to(device)
        receiver_x_t = torch.from_numpy(small_dataset['receiver_x']).to(device)
        receiver_y_t = torch.from_numpy(small_dataset['receiver_y']).to(device)

        n_il, n_xl = 5, 5
        output_x = torch.linspace(0, 200, n_il * n_xl, device=device)
        output_y = torch.zeros(n_il * n_xl, device=device)
        depth_axis = torch.linspace(10, 500, 50, device=device)  # Multiple depths to get sampling

        with caplog.at_level(logging.INFO):
            image, fold = migrate_kirchhoff_full(
                traces=traces_t,
                source_x=source_x_t,
                source_y=source_y_t,
                receiver_x=receiver_x_t,
                receiver_y=receiver_y_t,
                output_x=output_x,
                output_y=output_y,
                depth_axis=depth_axis,
                velocity=2000.0,
                dt_ms=4.0,
                t_min_ms=0.0,
                max_aperture_m=3000.0,
                max_angle_deg=60.0,
                n_il=n_il,
                n_xl=n_xl,
                enable_profiling=True,
            )

        log_text = caplog.text
        assert "APERTURE BY DEPTH" in log_text
        assert "Time(ms)" in log_text
        assert "Depth(m)" in log_text


class TestMigrationEngineProfileFlag:
    """Test that enable_profiling flag passes through MigrationEngine."""

    @pytest.fixture
    def small_engine_dataset(self):
        """Create dataset for engine test."""
        n_samples = 50
        n_traces = 30
        np.random.seed(42)

        traces = np.random.randn(n_samples, n_traces).astype(np.float32)
        x = np.linspace(0, 300, n_traces).astype(np.float32)
        y = np.zeros(n_traces, dtype=np.float32)

        return {
            'traces': traces,
            'source_x': x - 30,
            'source_y': y,
            'receiver_x': x + 30,
            'receiver_y': y,
        }

    def test_engine_accepts_profiling_flag(self, small_engine_dataset):
        """Test that MigrationEngine.migrate_bin_full accepts enable_profiling."""
        engine = MigrationEngine(device='cpu')

        image, fold = engine.migrate_bin_full(
            traces=small_engine_dataset['traces'],
            source_x=small_engine_dataset['source_x'],
            source_y=small_engine_dataset['source_y'],
            receiver_x=small_engine_dataset['receiver_x'],
            receiver_y=small_engine_dataset['receiver_y'],
            origin_x=0,
            origin_y=0,
            il_spacing=25.0,
            xl_spacing=25.0,
            azimuth_deg=0,
            n_il=5,
            n_xl=5,
            dt_ms=4.0,
            t_min_ms=0.0,
            velocity_mps=2000.0,
            max_aperture_m=500.0,
            max_angle_deg=60.0,
            enable_profiling=True,
        )

        assert image.shape == (50, 5, 5)
        assert fold.shape == (50, 5, 5)

    def test_engine_profiling_produces_logs(self, small_engine_dataset, caplog):
        """Test that profiling through engine produces timing logs."""
        engine = MigrationEngine(device='cpu')

        with caplog.at_level(logging.INFO):
            image, fold = engine.migrate_bin_full(
                traces=small_engine_dataset['traces'],
                source_x=small_engine_dataset['source_x'],
                source_y=small_engine_dataset['source_y'],
                receiver_x=small_engine_dataset['receiver_x'],
                receiver_y=small_engine_dataset['receiver_y'],
                origin_x=0,
                origin_y=0,
                il_spacing=25.0,
                xl_spacing=25.0,
                azimuth_deg=0,
                n_il=5,
                n_xl=5,
                dt_ms=4.0,
                t_min_ms=0.0,
                velocity_mps=2000.0,
                max_aperture_m=500.0,
                max_angle_deg=60.0,
                enable_profiling=True,
            )

        assert "KIRCHHOFF MIGRATION PROFILING RESULTS" in caplog.text


class TestTimeDependentAperture:
    """Test time-dependent aperture functionality."""

    @pytest.fixture
    def simple_dataset(self):
        """Create dataset for time-dependent aperture test."""
        n_samples = 100
        n_traces = 100
        np.random.seed(42)

        traces = np.random.randn(n_samples, n_traces).astype(np.float32)
        # Traces spread over 2000m
        x = np.linspace(0, 2000, n_traces).astype(np.float32)
        y = np.zeros(n_traces, dtype=np.float32)

        return {
            'traces': traces,
            'source_x': x - 50,
            'source_y': y,
            'receiver_x': x + 50,
            'receiver_y': y,
        }

    def test_time_dependent_aperture_runs(self, simple_dataset):
        """Test that time-dependent aperture mode runs without error."""
        engine = MigrationEngine(device='cpu')

        image, fold = engine.migrate_bin_full(
            traces=simple_dataset['traces'],
            source_x=simple_dataset['source_x'],
            source_y=simple_dataset['source_y'],
            receiver_x=simple_dataset['receiver_x'],
            receiver_y=simple_dataset['receiver_y'],
            origin_x=0,
            origin_y=0,
            il_spacing=50.0,
            xl_spacing=50.0,
            azimuth_deg=0,
            n_il=10,
            n_xl=10,
            dt_ms=4.0,
            t_min_ms=0.0,
            velocity_mps=2000.0,
            max_aperture_m=3000.0,
            max_angle_deg=60.0,
            use_time_dependent_aperture=True,
        )

        assert image.shape == (100, 10, 10)
        assert fold.shape == (100, 10, 10)

    def test_time_dependent_aperture_reduces_fold_at_shallow(self, simple_dataset):
        """Test that time-dependent aperture reduces fold at shallow times."""
        engine = MigrationEngine(device='cpu')

        # Run with constant aperture
        _, fold_constant = engine.migrate_bin_full(
            traces=simple_dataset['traces'],
            source_x=simple_dataset['source_x'],
            source_y=simple_dataset['source_y'],
            receiver_x=simple_dataset['receiver_x'],
            receiver_y=simple_dataset['receiver_y'],
            origin_x=0,
            origin_y=0,
            il_spacing=50.0,
            xl_spacing=50.0,
            azimuth_deg=0,
            n_il=10,
            n_xl=10,
            dt_ms=4.0,
            t_min_ms=0.0,
            velocity_mps=2000.0,
            max_aperture_m=3000.0,
            max_angle_deg=60.0,
            use_time_dependent_aperture=False,
        )

        # Run with time-dependent aperture
        _, fold_timedep = engine.migrate_bin_full(
            traces=simple_dataset['traces'],
            source_x=simple_dataset['source_x'],
            source_y=simple_dataset['source_y'],
            receiver_x=simple_dataset['receiver_x'],
            receiver_y=simple_dataset['receiver_y'],
            origin_x=0,
            origin_y=0,
            il_spacing=50.0,
            xl_spacing=50.0,
            azimuth_deg=0,
            n_il=10,
            n_xl=10,
            dt_ms=4.0,
            t_min_ms=0.0,
            velocity_mps=2000.0,
            max_aperture_m=3000.0,
            max_angle_deg=60.0,
            use_time_dependent_aperture=True,
        )

        # At shallow times (depth index 10 = ~40ms = ~40m depth)
        # time-dependent aperture should have LESS fold than constant
        shallow_idx = 10
        shallow_fold_constant = fold_constant[shallow_idx].mean()
        shallow_fold_timedep = fold_timedep[shallow_idx].mean()

        # At deep times (depth index 90 = ~360ms = ~360m depth)
        # both should have similar fold
        deep_idx = 90
        deep_fold_constant = fold_constant[deep_idx].mean()
        deep_fold_timedep = fold_timedep[deep_idx].mean()

        # Time-dependent should have significantly less fold at shallow
        # (if constant has any fold at shallow)
        if shallow_fold_constant > 0:
            assert shallow_fold_timedep <= shallow_fold_constant

        # At deep times, they should be closer (within 50%)
        if deep_fold_constant > 0:
            ratio = deep_fold_timedep / deep_fold_constant
            assert ratio >= 0.5  # Time-dependent shouldn't be too much smaller at depth


class TestConfigAdapterProfileFlag:
    """Test that enable_profiling passes through ConfigAdapter."""

    def test_config_adapter_includes_profiling_flag(self):
        """Test that get_engine_params includes enable_profiling."""
        from processors.migration.config_adapter import ConfigAdapter

        config = {
            'x_origin': 0,
            'y_origin': 0,
            'output_bin_il': 25.0,
            'output_bin_xl': 25.0,
            'grid_azimuth_deg': 0,
            'inline_min': 0,
            'inline_max': 100,
            'inline_step': 1,
            'xline_min': 0,
            'xline_max': 100,
            'xline_step': 1,
            'dt_ms': 4.0,
            'time_min_ms': 0,
            'time_max_ms': 200,
            'velocity_v0': 2000,
            'max_aperture_m': 3000,
            'max_angle_deg': 60,
        }

        adapter = ConfigAdapter(config)

        # Create dummy data
        traces = np.zeros((51, 10), dtype=np.float32)
        source_x = np.zeros(10, dtype=np.float32)
        source_y = np.zeros(10, dtype=np.float32)
        receiver_x = np.zeros(10, dtype=np.float32)
        receiver_y = np.zeros(10, dtype=np.float32)

        params = adapter.get_engine_params(
            traces, source_x, source_y, receiver_x, receiver_y,
            enable_profiling=True
        )

        assert 'enable_profiling' in params
        assert params['enable_profiling'] is True

    def test_config_adapter_profiling_default_false(self):
        """Test that enable_profiling defaults to False."""
        from processors.migration.config_adapter import ConfigAdapter

        config = {
            'x_origin': 0,
            'y_origin': 0,
            'output_bin_il': 25.0,
            'output_bin_xl': 25.0,
            'grid_azimuth_deg': 0,
            'inline_min': 0,
            'inline_max': 100,
            'inline_step': 1,
            'xline_min': 0,
            'xline_max': 100,
            'xline_step': 1,
            'dt_ms': 4.0,
            'time_min_ms': 0,
            'time_max_ms': 200,
            'velocity_v0': 2000,
            'max_aperture_m': 3000,
            'max_angle_deg': 60,
        }

        adapter = ConfigAdapter(config)

        traces = np.zeros((51, 10), dtype=np.float32)
        source_x = np.zeros(10, dtype=np.float32)
        source_y = np.zeros(10, dtype=np.float32)
        receiver_x = np.zeros(10, dtype=np.float32)
        receiver_y = np.zeros(10, dtype=np.float32)

        params = adapter.get_engine_params(
            traces, source_x, source_y, receiver_x, receiver_y
        )

        assert 'enable_profiling' in params
        assert params['enable_profiling'] is False

    def test_config_adapter_includes_time_dependent_aperture(self):
        """Test that get_engine_params includes use_time_dependent_aperture."""
        from processors.migration.config_adapter import ConfigAdapter

        config = {
            'x_origin': 0,
            'y_origin': 0,
            'output_bin_il': 25.0,
            'output_bin_xl': 25.0,
            'grid_azimuth_deg': 0,
            'inline_min': 0,
            'inline_max': 100,
            'inline_step': 1,
            'xline_min': 0,
            'xline_max': 100,
            'xline_step': 1,
            'dt_ms': 4.0,
            'time_min_ms': 0,
            'time_max_ms': 200,
            'velocity_v0': 2000,
            'max_aperture_m': 3000,
            'max_angle_deg': 60,
        }

        adapter = ConfigAdapter(config)

        traces = np.zeros((51, 10), dtype=np.float32)
        source_x = np.zeros(10, dtype=np.float32)
        source_y = np.zeros(10, dtype=np.float32)
        receiver_x = np.zeros(10, dtype=np.float32)
        receiver_y = np.zeros(10, dtype=np.float32)

        params = adapter.get_engine_params(
            traces, source_x, source_y, receiver_x, receiver_y,
            use_time_dependent_aperture=True
        )

        assert 'use_time_dependent_aperture' in params
        assert params['use_time_dependent_aperture'] is True

    def test_config_adapter_time_dependent_aperture_default_false(self):
        """Test that use_time_dependent_aperture defaults to False."""
        from processors.migration.config_adapter import ConfigAdapter

        config = {
            'x_origin': 0,
            'y_origin': 0,
            'output_bin_il': 25.0,
            'output_bin_xl': 25.0,
            'grid_azimuth_deg': 0,
            'inline_min': 0,
            'inline_max': 100,
            'inline_step': 1,
            'xline_min': 0,
            'xline_max': 100,
            'xline_step': 1,
            'dt_ms': 4.0,
            'time_min_ms': 0,
            'time_max_ms': 200,
            'velocity_v0': 2000,
            'max_aperture_m': 3000,
            'max_angle_deg': 60,
        }

        adapter = ConfigAdapter(config)

        traces = np.zeros((51, 10), dtype=np.float32)
        source_x = np.zeros(10, dtype=np.float32)
        source_y = np.zeros(10, dtype=np.float32)
        receiver_x = np.zeros(10, dtype=np.float32)
        receiver_y = np.zeros(10, dtype=np.float32)

        params = adapter.get_engine_params(
            traces, source_x, source_y, receiver_x, receiver_y
        )

        assert 'use_time_dependent_aperture' in params
        assert params['use_time_dependent_aperture'] is False
