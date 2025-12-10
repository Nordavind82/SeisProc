"""Tests for ConfigAdapter - wizard config to MigrationEngine mapping."""

import pytest
import numpy as np
from processors.migration.config_adapter import (
    ConfigAdapter,
    MigrationParams,
    create_adapter_from_wizard,
)


class TestMigrationParams:
    """Test MigrationParams dataclass."""

    def test_creation(self):
        """Test params can be created with all fields."""
        params = MigrationParams(
            origin_x=1000.0,
            origin_y=2000.0,
            il_spacing=25.0,
            xl_spacing=25.0,
            azimuth_deg=45.0,
            n_il=100,
            n_xl=100,
            dt_ms=4.0,
            t_min_ms=0.0,
            n_samples=1001,
            velocity_mps=3000.0,
            max_aperture_m=5000.0,
            max_angle_deg=60.0,
        )
        assert params.origin_x == 1000.0
        assert params.n_il == 100
        assert params.velocity_mps == 3000.0


class TestConfigAdapter:
    """Test ConfigAdapter wizard config mapping."""

    @pytest.fixture
    def wizard_config(self):
        """Standard wizard config fixture."""
        return {
            'name': 'Test Migration',
            'x_origin': 1000.0,
            'y_origin': 2000.0,
            'output_bin_il': 25.0,
            'output_bin_xl': 25.0,
            'grid_azimuth_deg': 45.0,
            'inline_min': 1,
            'inline_max': 100,
            'inline_step': 1,
            'xline_min': 1,
            'xline_max': 100,
            'xline_step': 1,
            'time_min_ms': 0,
            'time_max_ms': 4000,
            'dt_ms': 4.0,
            'velocity_type': 'constant',
            'velocity_v0': 3000.0,
            'max_aperture_m': 5000.0,
            'max_angle_deg': 60.0,
        }

    def test_adapter_creation(self, wizard_config):
        """Test adapter can be created from wizard config."""
        adapter = ConfigAdapter(wizard_config)
        assert adapter.config == wizard_config

    def test_factory_function(self, wizard_config):
        """Test factory function creates adapter."""
        adapter = create_adapter_from_wizard(wizard_config)
        assert isinstance(adapter, ConfigAdapter)

    def test_extract_origin(self, wizard_config):
        """Test origin extraction."""
        adapter = ConfigAdapter(wizard_config)
        params = adapter.params
        assert params.origin_x == 1000.0
        assert params.origin_y == 2000.0

    def test_extract_bin_sizes(self, wizard_config):
        """Test bin size extraction."""
        adapter = ConfigAdapter(wizard_config)
        params = adapter.params
        assert params.il_spacing == 25.0
        assert params.xl_spacing == 25.0

    def test_bin_size_fallback_to_input(self):
        """Test fallback from output_bin to input_bin."""
        config = {
            'input_bin_il': 50.0,
            'input_bin_xl': 50.0,
            # No output_bin values
        }
        adapter = ConfigAdapter(config)
        params = adapter.params
        assert params.il_spacing == 50.0
        assert params.xl_spacing == 50.0

    def test_bin_size_fallback_to_default(self):
        """Test fallback to 25.0 default."""
        config = {}  # Empty config
        adapter = ConfigAdapter(config)
        params = adapter.params
        assert params.il_spacing == 25.0
        assert params.xl_spacing == 25.0

    def test_extract_grid_azimuth(self, wizard_config):
        """Test azimuth extraction."""
        adapter = ConfigAdapter(wizard_config)
        params = adapter.params
        assert params.azimuth_deg == 45.0

    def test_compute_n_il_n_xl(self, wizard_config):
        """Test inline/crossline count computation."""
        adapter = ConfigAdapter(wizard_config)
        params = adapter.params
        # (100 - 1) / 1 + 1 = 100
        assert params.n_il == 100
        assert params.n_xl == 100

    def test_compute_n_il_n_xl_with_step(self):
        """Test count computation with step > 1."""
        config = {
            'inline_min': 1,
            'inline_max': 199,
            'inline_step': 2,  # Only odd inlines
            'xline_min': 1,
            'xline_max': 301,
            'xline_step': 3,  # Every 3rd crossline
        }
        adapter = ConfigAdapter(config)
        params = adapter.params
        # (199 - 1) / 2 + 1 = 100
        assert params.n_il == 100
        # (301 - 1) / 3 + 1 = 101
        assert params.n_xl == 101

    def test_extract_time_axis(self, wizard_config):
        """Test time axis extraction."""
        adapter = ConfigAdapter(wizard_config)
        params = adapter.params
        assert params.dt_ms == 4.0
        assert params.t_min_ms == 0.0
        # (4000 - 0) / 4 + 1 = 1001
        assert params.n_samples == 1001

    def test_extract_velocity(self, wizard_config):
        """Test velocity extraction."""
        adapter = ConfigAdapter(wizard_config)
        params = adapter.params
        assert params.velocity_mps == 3000.0

    def test_extract_migration_params(self, wizard_config):
        """Test aperture and angle extraction."""
        adapter = ConfigAdapter(wizard_config)
        params = adapter.params
        assert params.max_aperture_m == 5000.0
        assert params.max_angle_deg == 60.0


class TestGetEngineParams:
    """Test get_engine_params() method."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with standard config."""
        config = {
            'x_origin': 0.0,
            'y_origin': 0.0,
            'output_bin_il': 50.0,
            'output_bin_xl': 50.0,
            'grid_azimuth_deg': 0.0,
            'inline_min': 1,
            'inline_max': 100,
            'inline_step': 1,
            'xline_min': 1,
            'xline_max': 100,
            'xline_step': 1,
            'time_min_ms': 0,
            'time_max_ms': 2000,
            'dt_ms': 2.0,
            'velocity_v0': 3000.0,
            'max_aperture_m': 3000.0,
            'max_angle_deg': 45.0,
        }
        return ConfigAdapter(config)

    def test_returns_dict(self, adapter):
        """Test returns dictionary with all required keys."""
        traces = np.random.randn(1001, 100).astype(np.float32)
        source_x = np.random.randn(100).astype(np.float32)
        source_y = np.random.randn(100).astype(np.float32)
        receiver_x = np.random.randn(100).astype(np.float32)
        receiver_y = np.random.randn(100).astype(np.float32)

        params = adapter.get_engine_params(
            traces=traces,
            source_x=source_x,
            source_y=source_y,
            receiver_x=receiver_x,
            receiver_y=receiver_y,
        )

        # Check all required keys present
        required_keys = [
            'traces', 'source_x', 'source_y', 'receiver_x', 'receiver_y',
            'origin_x', 'origin_y', 'il_spacing', 'xl_spacing', 'azimuth_deg',
            'n_il', 'n_xl', 'dt_ms', 't_min_ms', 'velocity_mps',
            'max_aperture_m', 'max_angle_deg', 'normalize', 'min_fold',
            'progress_callback'
        ]
        for key in required_keys:
            assert key in params, f"Missing key: {key}"

    def test_passes_trace_data(self, adapter):
        """Test trace data passed through correctly."""
        traces = np.random.randn(1001, 100).astype(np.float32)
        source_x = np.random.randn(100).astype(np.float32)

        params = adapter.get_engine_params(
            traces=traces,
            source_x=source_x,
            source_y=np.zeros(100),
            receiver_x=np.zeros(100),
            receiver_y=np.zeros(100),
        )

        assert np.array_equal(params['traces'], traces)
        assert np.array_equal(params['source_x'], source_x)

    def test_applies_extracted_params(self, adapter):
        """Test extracted params applied correctly."""
        params = adapter.get_engine_params(
            traces=np.zeros((1001, 10)),
            source_x=np.zeros(10),
            source_y=np.zeros(10),
            receiver_x=np.zeros(10),
            receiver_y=np.zeros(10),
        )

        assert params['il_spacing'] == 50.0
        assert params['xl_spacing'] == 50.0
        assert params['n_il'] == 100
        assert params['n_xl'] == 100
        assert params['velocity_mps'] == 3000.0
        assert params['max_aperture_m'] == 3000.0

    def test_optional_params(self, adapter):
        """Test optional parameters passed through."""
        def progress_fn(pct, msg):
            pass

        params = adapter.get_engine_params(
            traces=np.zeros((1001, 10)),
            source_x=np.zeros(10),
            source_y=np.zeros(10),
            receiver_x=np.zeros(10),
            receiver_y=np.zeros(10),
            normalize=False,
            min_fold=3,
            progress_callback=progress_fn,
        )

        assert params['normalize'] is False
        assert params['min_fold'] == 3
        assert params['progress_callback'] is progress_fn


class TestValidation:
    """Test validate() method."""

    def test_valid_config(self):
        """Test validation passes for valid config."""
        config = {
            'x_origin': 0.0,
            'y_origin': 0.0,
            'output_bin_il': 25.0,
            'output_bin_xl': 25.0,
            'inline_min': 1,
            'inline_max': 100,
            'inline_step': 1,
            'xline_min': 1,
            'xline_max': 100,
            'xline_step': 1,
            'time_min_ms': 0,
            'time_max_ms': 4000,
            'dt_ms': 4.0,
            'velocity_v0': 3000.0,
            'max_aperture_m': 5000.0,
            'max_angle_deg': 60.0,
        }
        adapter = ConfigAdapter(config)
        is_valid, msg = adapter.validate()
        assert is_valid, msg

    def test_invalid_velocity(self):
        """Test validation fails for zero velocity."""
        config = {
            'velocity_v0': 0.0,  # Invalid
        }
        adapter = ConfigAdapter(config)
        is_valid, msg = adapter.validate()
        assert not is_valid
        assert 'velocity' in msg.lower()

    def test_unusual_velocity_warning(self):
        """Test unusual velocity gets flagged."""
        config = {
            'velocity_v0': 500.0,  # Too slow
        }
        adapter = ConfigAdapter(config)
        is_valid, msg = adapter.validate()
        assert not is_valid
        assert 'unusual' in msg.lower() or 'velocity' in msg.lower()

    def test_invalid_grid_size(self):
        """Test validation fails for zero grid size."""
        config = {
            'inline_min': 100,
            'inline_max': 10,  # max < min -> n_il = 0
            'inline_step': 1,
        }
        adapter = ConfigAdapter(config)
        is_valid, msg = adapter.validate()
        assert not is_valid
        assert 'inline' in msg.lower()

    def test_grid_too_large(self):
        """Test validation fails for excessively large grid."""
        config = {
            'inline_min': 1,
            'inline_max': 100000,  # Very large
            'inline_step': 1,
            'xline_min': 1,
            'xline_max': 100000,
            'xline_step': 1,
        }
        adapter = ConfigAdapter(config)
        is_valid, msg = adapter.validate()
        assert not is_valid
        assert 'large' in msg.lower()

    def test_invalid_dt(self):
        """Test validation fails for zero dt."""
        config = {
            'dt_ms': 0.0,  # Invalid
        }
        adapter = ConfigAdapter(config)
        is_valid, msg = adapter.validate()
        assert not is_valid
        assert 'interval' in msg.lower() or 'dt' in msg.lower()

    def test_invalid_angle(self):
        """Test validation fails for angle > 90."""
        config = {
            'max_angle_deg': 95.0,  # > 90
        }
        adapter = ConfigAdapter(config)
        is_valid, msg = adapter.validate()
        assert not is_valid
        assert 'angle' in msg.lower()


class TestGetSummary:
    """Test get_summary() method."""

    def test_returns_string(self):
        """Test summary is a readable string."""
        config = {
            'x_origin': 1000.0,
            'y_origin': 2000.0,
            'output_bin_il': 25.0,
            'output_bin_xl': 25.0,
            'grid_azimuth_deg': 45.0,
            'inline_min': 1,
            'inline_max': 100,
            'inline_step': 1,
            'xline_min': 1,
            'xline_max': 100,
            'xline_step': 1,
            'time_min_ms': 0,
            'time_max_ms': 4000,
            'dt_ms': 4.0,
            'velocity_v0': 3000.0,
            'max_aperture_m': 5000.0,
            'max_angle_deg': 60.0,
        }
        adapter = ConfigAdapter(config)
        summary = adapter.get_summary()

        assert isinstance(summary, str)
        assert '100' in summary  # n_il or n_xl
        assert '3000' in summary  # velocity
        assert '5000' in summary  # aperture
        assert '25' in summary  # bin size

    def test_summary_contains_all_sections(self):
        """Test summary contains all important info."""
        config = {
            'velocity_v0': 2500.0,
        }
        adapter = ConfigAdapter(config)
        summary = adapter.get_summary()

        assert 'grid' in summary.lower()
        assert 'origin' in summary.lower()
        assert 'velocity' in summary.lower()
        assert 'aperture' in summary.lower()


class TestIntegrationWithMigrationEngine:
    """Test that adapter produces valid params for MigrationEngine."""

    def test_params_match_migrate_bin_full_signature(self):
        """Test params match MigrationEngine.migrate_bin_full() args."""
        from processors.migration.migration_engine import MigrationEngine
        import inspect

        # Get migrate_bin_full signature
        sig = inspect.signature(MigrationEngine.migrate_bin_full)
        expected_params = set(sig.parameters.keys()) - {'self'}

        # Create adapter and get params
        config = {
            'x_origin': 0.0,
            'y_origin': 0.0,
            'output_bin_il': 25.0,
            'output_bin_xl': 25.0,
            'inline_min': 1,
            'inline_max': 100,
            'inline_step': 1,
            'xline_min': 1,
            'xline_max': 100,
            'xline_step': 1,
            'time_min_ms': 0,
            'time_max_ms': 4000,
            'dt_ms': 4.0,
            'velocity_v0': 3000.0,
        }
        adapter = ConfigAdapter(config)
        params = adapter.get_engine_params(
            traces=np.zeros((1001, 10)),
            source_x=np.zeros(10),
            source_y=np.zeros(10),
            receiver_x=np.zeros(10),
            receiver_y=np.zeros(10),
        )

        # All adapter params should be in method signature
        adapter_params = set(params.keys())
        for key in adapter_params:
            assert key in expected_params, f"Adapter param '{key}' not in migrate_bin_full signature"
