# 3D FKK Filter Implementation - Task List

## Project Overview

Implement a universal 3D FKK (Frequency-Wavenumber-Wavenumber) filter for SeisProc with GPU acceleration, memory-efficient processing for large 3D volumes, and interactive visualization following existing architectural patterns.

**Integration Points:**
- Extend `BaseProcessor` pattern from `processors/base_processor.py`
- Use `DeviceManager` from `processors/gpu/device_manager.py`
- Follow `FKDesignerDialog` UI patterns from `views/fk_designer_dialog.py`
- Integrate with `ChunkedProcessor` patterns for memory management
- Use existing parallel processing framework from `utils/parallel_processing/`

---

## Phase 1: Core Data Structures & Models

### 1.1 3D Volume Data Model
**File:** `models/seismic_volume.py`

- [ ] Create `SeismicVolume` class for 3D data (nt, nx, ny)
  - Immutable container similar to `SeismicData`
  - Store: traces_3d (numpy array), geometry (dx, dy, dt), headers
  - Properties: n_samples, n_xlines, n_inlines, shape
  - Methods: `get_inline(idx)`, `get_xline(idx)`, `get_time_slice(idx)`
  - Support coordinate systems: inline/xline or x/y meters

- [ ] Create `LazySeismicVolume` class for memory-mapped 3D volumes
  - Extend patterns from `models/lazy_seismic_data.py`
  - Zarr-backed storage with chunking strategy (chunks along all 3 axes)
  - Methods: `get_subcube(t_range, x_range, y_range)`
  - Memory footprint target: constant ~20MB regardless of volume size

- [ ] Add volume geometry validation
  - Check regular grid spacing (or handle irregular with interpolation flag)
  - Validate Nyquist limits for all three dimensions
  - Warn about spatial aliasing risks

### 1.2 3D FKK Configuration Model
**File:** `models/fkk_config.py`

- [ ] Create `FKKFilterConfig` dataclass extending `FKFilterConfig` patterns
  ```python
  @dataclass
  class FKKFilterConfig:
      name: str
      filter_type: str  # 'velocity_cone', 'polygon', 'dip', 'radial'

      # Velocity cone parameters
      v_min: Optional[float]
      v_max: Optional[float]
      azimuth_min: float = 0.0    # degrees, 0-360
      azimuth_max: float = 360.0

      # Polygon parameters (list of 3D polygons in FKK space)
      polygons: List[FKKPolygon] = field(default_factory=list)

      # Dip parameters (slowness-based)
      px_min: Optional[float]  # s/m in x-direction
      px_max: Optional[float]
      py_min: Optional[float]  # s/m in y-direction
      py_max: Optional[float]

      # Common parameters
      taper_width: float = 0.1  # fraction of filter boundary
      taper_type: str = 'cosine'  # 'cosine', 'gaussian', 'butterworth'
      mode: str = 'reject'  # 'pass' or 'reject'

      # Processing parameters
      padding_factor: float = 1.0  # 1.0 = no padding, 2.0 = double
      preserve_dc: bool = True

      # Units
      coordinate_units: str = 'meters'
  ```

- [ ] Create `FKKPolygon` class for polygon-based filter zones
  - Store vertices in (f, kx, ky) space
  - Support pass/reject mode per polygon
  - Methods: `contains_point()`, `to_3d_mask()`, `interpolate_between_slices()`

- [ ] Create `FKKConfigManager` class
  - Save/load configs to `~/.denoise_app/fkk_configs/`
  - Built-in presets: Ground Roll 3D, Linear Noise, Coherent Noise
  - Import/export JSON format

### 1.3 FKK Axis Utilities
**File:** `utils/fkk_axes.py`

- [ ] Implement axis computation functions
  ```python
  def compute_frequency_axis(n_samples: int, dt: float) -> np.ndarray:
      """Compute frequency axis (0 to f_nyquist, positive only)."""

  def compute_wavenumber_axis(n_traces: int, dx: float) -> np.ndarray:
      """Compute centered wavenumber axis (-k_nyquist to +k_nyquist)."""

  def compute_velocity_volume(f_axis, kx_axis, ky_axis) -> np.ndarray:
      """Compute velocity at each (f, kx, ky) point: v = f / sqrt(kx² + ky²)."""

  def compute_azimuth_volume(kx_axis, ky_axis) -> np.ndarray:
      """Compute azimuth at each (kx, ky) point: azimuth = atan2(ky, kx)."""

  def compute_aliasing_velocity(f: float, dx: float, dy: float) -> Tuple[float, float]:
      """Compute aliasing velocities in x and y directions."""
  ```

- [ ] Add unit conversion utilities for FKK domain
  - Velocity: m/s ↔ ft/s
  - Wavenumber: 1/m ↔ 1/ft
  - Slowness: s/m ↔ s/ft

---

## Phase 2: GPU Compute Engine

### 2.1 GPU 3D FFT Module
**File:** `processors/gpu/fft3d_gpu.py`

- [ ] Create `FFT3DGPU` class using CuPy/PyTorch
  ```python
  class FFT3DGPU:
      def __init__(self, shape: Tuple[int, int, int], device_manager: DeviceManager):
          self.shape = shape
          self.device = device_manager
          self._plan = None  # Cached FFT plan

      def forward(self, data: np.ndarray) -> np.ndarray:
          """3D FFT: (t, x, y) → (f, kx, ky)"""

      def inverse(self, spectrum: np.ndarray) -> np.ndarray:
          """3D IFFT: (f, kx, ky) → (t, x, y)"""

      def _create_plan(self):
          """Pre-compute FFT plan for efficiency."""
  ```

- [ ] Implement FFT plan caching
  - Cache plans by shape (reuse for same-size volumes)
  - LRU eviction for plan cache (limit: 5 plans)
  - Memory: plans can be large, track in DeviceManager

- [ ] Add CPU fallback using `scipy.fft.fftn`
  - Automatic fallback on GPU OOM or unavailability
  - Same interface as GPU version

- [ ] Benchmark and optimize
  - Target: 256³ < 100ms, 512³ < 500ms
  - Test with float32 and float64
  - Profile: FFT time vs transfer time vs filter application

### 2.2 GPU FKK Filter Processor
**File:** `processors/fkk_filter_gpu.py`

- [ ] Create `FKKFilterGPU` class extending `BaseProcessor`
  ```python
  class FKKFilterGPU(BaseProcessor):
      def __init__(self, config: FKKFilterConfig, device_manager: DeviceManager = None):
          self.config = config
          self.device = device_manager or get_device_manager()
          self._fft_engine = None
          self._filter_cache = {}

      def process(self, volume: SeismicVolume) -> SeismicVolume:
          """Apply FKK filter to 3D volume."""

      def compute_fkk_spectrum(self, volume: SeismicVolume) -> np.ndarray:
          """Compute FKK spectrum for visualization."""

      def _build_filter_mask(self, shape, geometry) -> np.ndarray:
          """Build 3D filter mask on GPU."""

      def _apply_taper(self, mask: np.ndarray) -> np.ndarray:
          """Apply smooth taper to filter boundaries."""
  ```

- [ ] Implement velocity cone filter on GPU
  ```python
  def _create_velocity_cone_mask(self, f_axis, kx_axis, ky_axis):
      # Compute velocity at each point: v = f / sqrt(kx² + ky²)
      # Create mask based on v_min, v_max, azimuth_range
      # Apply cosine taper at boundaries
  ```

- [ ] Implement polygon filter on GPU
  - Convert polygon vertices to 3D mask
  - Use GPU-accelerated point-in-polygon test
  - Support multiple polygons with AND/OR combination

- [ ] Implement dip filter on GPU
  - Filter based on slowness (px, py)
  - Relationship: kx = f * px, ky = f * py

- [ ] Implement radial filter on GPU
  - Filter based on k-magnitude: k = sqrt(kx² + ky²)

- [ ] Add serialization for multiprocessing
  ```python
  def to_dict(self) -> Dict[str, Any]:
      return {'config': asdict(self.config), 'type': 'FKKFilterGPU'}

  @classmethod
  def from_dict(cls, config: Dict) -> 'FKKFilterGPU':
      return cls(FKKFilterConfig(**config['config']))
  ```

### 2.3 Taper Functions
**File:** `processors/gpu/tapers.py`

- [ ] Implement taper functions on GPU
  ```python
  def cosine_taper(x, width):
      """Cosine (Tukey) taper: 0.5 * (1 - cos(π * x / width))"""

  def gaussian_taper(x, sigma):
      """Gaussian taper: exp(-0.5 * (x / sigma)²)"""

  def butterworth_taper(x, cutoff, order):
      """Butterworth rolloff: 1 / (1 + (x / cutoff)^(2*order))"""
  ```

- [ ] Create taper preview utility for UI
  - Generate 1D taper profile for visualization
  - Show taper effect on filter boundary

---

## Phase 3: Memory Management for Large Volumes

### 3.1 Chunked 3D Processing
**File:** `processors/chunked_fkk_processor.py`

- [ ] Design chunking strategy for 3D FKK
  - **Challenge:** FKK requires full volume for proper transform
  - **Strategy 1:** Process in overlapping 3D tiles with blend
  - **Strategy 2:** Process frequency slices independently (partial)
  - **Strategy 3:** Downsample → design filter → upsample mask → apply in chunks

- [ ] Implement tile-based processing with overlap
  ```python
  class ChunkedFKKProcessor:
      def __init__(self, chunk_shape: Tuple[int, int, int],
                   overlap_percent: float = 0.2):
          self.chunk_shape = chunk_shape
          self.overlap = overlap_percent

      def process_volume(self, input_zarr, output_zarr, filter_mask,
                        progress_callback=None):
          # Divide volume into overlapping tiles
          # Process each tile with FFT → filter → IFFT
          # Blend overlapping regions with raised-cosine window
          # Write to output Zarr
  ```

- [ ] Implement overlap-add/overlap-save for boundary handling
  - Raised-cosine blending windows
  - Artifact detection at tile boundaries
  - Quality metrics: boundary discontinuity measure

- [ ] Add memory estimation utility
  ```python
  def estimate_memory_requirements(volume_shape, dtype, padding_factor):
      """Estimate peak GPU memory for FKK processing."""
      # FFT of padded volume
      # Complex spectrum storage
      # Filter mask storage
      # Intermediate buffers
  ```

### 3.2 GPU Memory Management
**File:** `processors/gpu/memory_manager_3d.py`

- [ ] Extend `DeviceManager` for 3D volume operations
  ```python
  class VolumeMemoryManager:
      def __init__(self, device_manager: DeviceManager):
          self.device = device_manager

      def can_fit_volume(self, shape, dtype) -> bool:
          """Check if volume fits in available GPU memory."""

      def optimal_chunk_shape(self, volume_shape, available_memory) -> Tuple:
          """Calculate optimal chunk shape for available memory."""

      def estimate_fkk_memory(self, shape) -> int:
          """Estimate memory for full FKK round-trip."""
  ```

- [ ] Implement dynamic chunking based on available memory
  - Query available GPU memory at runtime
  - Adjust chunk size dynamically
  - Reserve headroom (80% of available)

- [ ] Add OOM recovery with automatic chunking fallback
  ```python
  def process_with_fallback(self, volume, filter_mask):
      try:
          return self._process_full(volume, filter_mask)
      except OutOfMemoryError:
          logger.warning("GPU OOM, falling back to chunked processing")
          return self._process_chunked(volume, filter_mask)
  ```

### 3.3 Zarr Integration for 3D Volumes
**File:** `utils/zarr_volume_manager.py`

- [ ] Create Zarr storage manager for 3D volumes
  ```python
  class ZarrVolumeManager:
      def create_volume(self, path, shape, dtype, chunks='auto'):
          """Create new Zarr volume with optimal chunking."""

      def optimal_chunks_3d(self, shape, target_chunk_mb=64):
          """Calculate optimal chunk shape for 3D seismic."""
          # Balance: access patterns (time slices, inlines, xlines)
          # Typical: (64, 64, 64) or (128, 32, 32) for seismic

      def copy_volume_chunked(self, src, dst, transform_fn=None):
          """Copy volume with optional transform, chunk by chunk."""
  ```

- [ ] Implement efficient chunk access patterns
  - Time-slice access: chunks along time axis
  - Inline access: chunks along inline axis
  - FKK access: full volume (streaming if needed)

---

## Phase 4: Parallel Batch Processing

### 4.1 Extend Parallel Processing Framework
**File:** `utils/parallel_processing/volume_coordinator.py`

- [ ] Create `VolumeProcessingCoordinator` for 3D batch jobs
  ```python
  class VolumeProcessingCoordinator:
      def __init__(self, config: VolumeProcessingConfig):
          self.config = config
          self.progress_queue = mp.Queue()

      def process_volumes(self, input_volumes: List[Path],
                         filter_config: FKKFilterConfig,
                         output_dir: Path) -> BatchResult:
          # Partition volumes across workers
          # Each worker processes complete volumes
          # Track progress per volume
  ```

- [ ] Implement volume partitioning strategy
  - **Option A:** One volume per worker (simple, good for many small volumes)
  - **Option B:** Share large volumes across workers (chunk-based)
  - **Option C:** Hybrid based on volume size

- [ ] Add multi-GPU support
  ```python
  def distribute_across_gpus(self, volumes, gpu_ids):
      # Assign volumes to GPUs round-robin
      # Or load-balance by volume size
      # Each worker binds to specific GPU
  ```

### 4.2 Worker Process for 3D Volumes
**File:** `utils/parallel_processing/volume_worker.py`

- [ ] Create worker process for volume processing
  ```python
  def process_volume_worker(task: VolumeTask, progress_queue: mp.Queue):
      # Set GPU device (if multi-GPU)
      # Load volume (or chunk)
      # Create FKKFilterGPU processor
      # Process and write output
      # Report progress
  ```

- [ ] Implement GPU affinity for workers
  - Pin worker to specific GPU
  - Prevent GPU contention between workers
  - Handle GPU availability dynamically

### 4.3 Progress and Monitoring
**File:** `utils/parallel_processing/volume_monitor.py`

- [ ] Create monitoring system for batch processing
  ```python
  class VolumeProcessingMonitor:
      def __init__(self):
          self.volumes_total = 0
          self.volumes_completed = 0
          self.current_volume_progress = {}
          self.gpu_utilization = {}

      def update(self, worker_id, volume_id, progress, gpu_stats):
          # Update progress tracking
          # Emit signals for UI update
  ```

- [ ] Add GPU utilization tracking
  - Memory usage per GPU
  - Compute utilization
  - Thermal throttling warnings

---

## Phase 5: UI Components

### 5.1 3D Volume Viewer
**File:** `views/volume_viewer.py`

- [ ] Create `VolumeViewer` widget with orthogonal slices
  ```python
  class VolumeViewer(QWidget):
      # Three synchronized PyQtGraph ImageViews
      # - Time slice (inline × xline at fixed time)
      # - Inline section (time × xline at fixed inline)
      # - Crossline section (time × inline at fixed xline)

      def __init__(self):
          self.time_slice_view = pg.ImageView()
          self.inline_view = pg.ImageView()
          self.xline_view = pg.ImageView()
          self.cursor_position = (0, 0, 0)  # (t, x, y)

      def set_volume(self, volume: SeismicVolume):
          """Load volume and display initial slices."""

      def set_slice_indices(self, t_idx, x_idx, y_idx):
          """Update all three slice views."""

      def _on_cursor_moved(self, view, pos):
          """Synchronize cursor across all views."""
  ```

- [ ] Add lazy loading for large volumes
  - Load only visible slices
  - Pre-fetch adjacent slices
  - Cache recent slices (LRU, configurable size)

- [ ] Implement synchronized cursors
  - Click in one view updates cursor lines in all views
  - Cursor position displayed in status bar
  - Keyboard navigation: arrow keys, page up/down

- [ ] Add toggle: Input / Filtered / Difference
  - Side-by-side or overlay modes
  - Difference amplification slider
  - Quality metrics display

### 5.2 3D FKK Spectrum Viewer
**File:** `views/fkk_spectrum_viewer.py`

- [ ] Create `FKKSpectrumViewer` widget
  ```python
  class FKKSpectrumViewer(QWidget):
      # Three orthogonal slice views of FKK spectrum
      # - f-kx slice (at fixed ky)
      # - f-ky slice (at fixed kx)
      # - kx-ky slice (at fixed f)

      def __init__(self):
          self.f_kx_view = pg.ImageView()
          self.f_ky_view = pg.ImageView()
          self.kx_ky_view = pg.ImageView()

      def set_spectrum(self, spectrum: np.ndarray, axes: FKKAxes):
          """Load FKK spectrum and initialize views."""

      def add_filter_overlay(self, filter_mask: np.ndarray):
          """Overlay filter boundary on spectrum views."""
  ```

- [ ] Add spectrum display options
  - Amplitude: linear, log (dB), power
  - Smoothing: Gaussian filter (0-5 levels)
  - Gain: logarithmic slider
  - Colormap: Hot, Viridis, Turbo, Seismic

- [ ] Implement overlay system
  - Velocity isolines (v = f/k)
  - Aliasing boundaries (v_alias = f * dx)
  - Current filter boundary
  - Custom annotation support

- [ ] Add axis labels and scale bars
  - Frequency axis: Hz
  - Wavenumber axes: 1/m or cycles/m
  - Velocity contour labels

### 5.3 Interactive Filter Designer
**File:** `views/fkk_designer_dialog.py`

- [ ] Create `FKKDesignerDialog` main window
  ```python
  class FKKDesignerDialog(QDialog):
      def __init__(self, volume: SeismicVolume, parent=None):
          # Layout:
          # Left: Volume viewer (input/filtered/difference)
          # Right: FKK spectrum viewer (with filter overlay)
          # Bottom: Filter parameter controls
          # Footer: Status bar with metrics
  ```

- [ ] Implement filter type selection panel
  ```python
  class FilterTypePanel(QWidget):
      # Radio buttons: Velocity Cone, Polygon, Dip, Radial
      # Stacked widget with type-specific controls

      # Velocity Cone:
      #   - v_min, v_max spinboxes with units
      #   - Azimuth range: min, max (0-360°)
      #   - Visual: cone preview in 3D mini-view

      # Polygon:
      #   - Draw mode toggle
      #   - Polygon list with pass/reject toggle
      #   - Vertex editing

      # Dip:
      #   - px_min, px_max, py_min, py_max
      #   - Convert to/from velocity display

      # Radial:
      #   - k_min, k_max sliders
  ```

- [ ] Implement polygon drawing tools
  ```python
  class PolygonDrawingTool:
      # Click to add vertex
      # Right-click to close polygon
      # Drag vertex to move
      # Double-click vertex to delete
      # Shift+click to add vertex on edge

      def start_polygon(self):
      def add_vertex(self, pos):
      def close_polygon(self):
      def edit_vertex(self, idx, new_pos):
      def delete_polygon(self, idx):
  ```

- [ ] Implement velocity handle dragging
  - Drag v_min line to adjust minimum velocity
  - Drag v_max line to adjust maximum velocity
  - Visual feedback: line thickness on hover
  - Snap to nice values (optional)

- [ ] Add real-time preview system
  ```python
  class PreviewManager:
      def __init__(self, volume, spectrum_viewer, volume_viewer):
          self.preview_timer = QTimer()
          self.preview_timer.timeout.connect(self._update_preview)
          self.preview_decimation = 4  # Subsample for speed

      def request_preview(self):
          """Debounced preview request."""
          self.preview_timer.start(100)  # 100ms debounce

      def _update_preview(self):
          # Apply filter to decimated volume
          # Update viewers
          # Show compute time in status
  ```

- [ ] Add taper controls
  - Taper type: Cosine, Gaussian, Butterworth
  - Taper width: slider (0.01 - 0.5)
  - Preview: 1D taper profile plot
  - Butterworth order (if selected)

- [ ] Implement aliasing warning system
  - Show aliasing boundaries on spectrum
  - Warn if filter crosses aliasing zone
  - Color-code aliased regions

### 5.4 Preset Management UI
**File:** `views/fkk_preset_panel.py`

- [ ] Create preset management panel
  ```python
  class FKKPresetPanel(QWidget):
      # Preset list with built-in and user presets
      # Load/Save/Delete buttons
      # Preset preview thumbnail
      # Import/Export to file

      def __init__(self):
          self.preset_list = QListWidget()
          self.load_btn = QPushButton("Load")
          self.save_btn = QPushButton("Save")

      def _load_built_in_presets(self):
          # Ground Roll 3D: low velocity cone
          # Linear Noise: narrow azimuth fan
          # Coherent Noise: specific velocity/azimuth
          # Reflection Pass: high velocity cone
  ```

- [ ] Add preset preview generation
  - Generate 2D slice preview of filter
  - Show key parameters in tooltip
  - Quick-apply on double-click

### 5.5 Batch Processing Dialog
**File:** `views/fkk_batch_dialog.py`

- [ ] Create batch processing configuration dialog
  ```python
  class FKKBatchDialog(QDialog):
      # Input: List of volumes to process
      # Filter: Current or saved configuration
      # Output: Directory selection
      # Options:
      #   - Worker count
      #   - GPU selection (if multi-GPU)
      #   - Chunk size (if memory-limited)
      # Progress: Per-volume progress bars

      def __init__(self, volumes: List[SeismicVolume],
                   config: FKKFilterConfig):
          ...

      def _start_processing(self):
          # Validate disk space
          # Launch coordinator
          # Show progress dialog
  ```

- [ ] Implement progress visualization
  - Overall progress bar
  - Per-worker progress bars
  - GPU memory/utilization graphs
  - ETA estimation
  - Cancel button with graceful shutdown

---

## Phase 6: Integration with Existing App

### 6.1 Main Window Integration
**File:** `main_window.py` (extend existing)

- [ ] Add 3D FKK menu items
  ```python
  # In _create_menus():
  fkk_menu = self.menuBar().addMenu("3D FKK")
  fkk_menu.addAction("Design 3D FKK Filter...", self._open_fkk_designer)
  fkk_menu.addAction("Apply Saved FKK Config...", self._apply_fkk_config)
  fkk_menu.addAction("Batch Process Volumes...", self._open_fkk_batch)
  fkk_menu.addSeparator()
  fkk_menu.addAction("FKK Presets...", self._manage_fkk_presets)
  ```

- [ ] Add toolbar buttons for FKK operations
  - "Design FKK" button with icon
  - "Apply FKK" dropdown with recent configs

- [ ] Implement volume loading workflow
  ```python
  def _load_3d_volume(self):
      # File dialog for SEG-Y 3D or Zarr volume
      # Progress dialog for large files
      # Add to dataset navigator
      # Switch to 3D view mode
  ```

### 6.2 Control Panel Integration
**File:** `views/control_panel.py` (extend existing)

- [ ] Add FKK section to control panel
  ```python
  # FKK Filter Section
  self.fkk_group = QGroupBox("3D FKK Filter")
  self.fkk_config_combo = QComboBox()  # Saved configurations
  self.fkk_design_btn = QPushButton("Design...")
  self.fkk_apply_btn = QPushButton("Apply")
  self.fkk_preview_checkbox = QCheckBox("Auto-preview")
  ```

- [ ] Connect to existing processing pipeline
  - Emit `fkk_process_requested` signal
  - Handle in MainWindow like other processors

### 6.3 Dataset Navigator Integration
**File:** `models/dataset_navigator.py` (extend existing)

- [ ] Add support for 3D volumes in dataset navigator
  ```python
  def add_volume(self, volume: SeismicVolume, name: str):
      """Add 3D volume to navigator."""

  def get_current_volume(self) -> Optional[SeismicVolume]:
      """Get currently selected volume."""

  @property
  def is_3d_mode(self) -> bool:
      """Check if current dataset is 3D volume."""
  ```

- [ ] Implement view mode switching
  - 2D gather mode (existing)
  - 3D volume mode (new)
  - UI adapts based on mode

### 6.4 Storage Manager Integration
**File:** `utils/storage_manager.py` (extend existing)

- [ ] Add volume storage methods
  ```python
  def create_volume_session(self, volume_id: str) -> Path:
      """Create storage session for 3D volume."""

  def save_volume_zarr(self, volume: SeismicVolume, session_path: Path):
      """Save volume to Zarr format."""

  def load_volume_zarr(self, session_path: Path) -> LazySeismicVolume:
      """Load volume from Zarr."""
  ```

- [ ] Add disk space validation for 3D volumes
  - Estimate: input + output + temp space
  - Warn if insufficient
  - Suggest compression options

---

## Phase 7: Testing & Validation

### 7.1 Unit Tests
**Directory:** `tests/test_fkk/`

- [ ] Test FKK transform accuracy
  ```python
  # test_fkk_transform.py
  def test_fkk_roundtrip():
      """FFT → IFFT should recover original."""

  def test_fkk_axes_computation():
      """Verify frequency and wavenumber axes."""

  def test_fkk_parseval():
      """Energy conservation: time domain = frequency domain."""
  ```

- [ ] Test filter creation
  ```python
  # test_fkk_filters.py
  def test_velocity_cone_filter():
      """Verify velocity cone passes correct events."""

  def test_polygon_filter():
      """Verify polygon mask creation."""

  def test_taper_smoothness():
      """Verify taper creates smooth transitions."""
  ```

- [ ] Test GPU/CPU equivalence
  ```python
  # test_gpu_cpu_equivalence.py
  def test_fkk_gpu_cpu_match():
      """GPU and CPU results should match within tolerance."""
  ```

### 7.2 Integration Tests
**Directory:** `tests/test_fkk/`

- [ ] Test with synthetic data
  ```python
  # test_fkk_synthetic.py
  def create_synthetic_volume_with_events():
      """Create volume with known velocity events."""
      # Plane waves at known velocities and azimuths
      # Verify filter attenuates correct events

  def test_ground_roll_removal():
      """Verify low-velocity noise rejection."""

  def test_reflection_preservation():
      """Verify high-velocity events preserved."""
  ```

- [ ] Test memory management
  ```python
  # test_fkk_memory.py
  def test_chunked_processing_large_volume():
      """Process volume larger than GPU memory."""

  def test_memory_not_exceeded():
      """Monitor peak memory during processing."""
  ```

- [ ] Test parallel processing
  ```python
  # test_fkk_parallel.py
  def test_batch_processing_multiple_volumes():
      """Process batch of volumes in parallel."""

  def test_multi_gpu_distribution():
      """Verify work distributed across GPUs."""
  ```

### 7.3 Performance Benchmarks
**File:** `tests/benchmarks/benchmark_fkk.py`

- [ ] Create benchmark suite
  ```python
  BENCHMARK_SIZES = [
      (128, 128, 128),   # Small
      (256, 256, 256),   # Medium
      (512, 512, 256),   # Large
      (512, 512, 512),   # Very large
  ]

  def benchmark_fkk_transform(size):
      """Benchmark FFT + filter + IFFT."""

  def benchmark_gpu_vs_cpu(size):
      """Compare GPU and CPU performance."""

  def benchmark_chunked_overhead(size, chunk_size):
      """Measure chunking overhead."""
  ```

- [ ] Performance targets
  - 256³: < 100ms (GPU), < 2s (CPU)
  - 512³: < 500ms (GPU), < 15s (CPU)
  - Chunked overhead: < 20% vs full-volume

### 7.4 UI Tests
**Directory:** `tests/test_fkk/`

- [ ] Test viewer functionality
  ```python
  # test_fkk_viewers.py
  def test_volume_viewer_slice_sync():
      """Verify slice synchronization."""

  def test_spectrum_viewer_overlay():
      """Verify filter overlay rendering."""

  def test_polygon_drawing():
      """Test polygon creation and editing."""
  ```

---

## Phase 8: Documentation & Polish

### 8.1 User Documentation
**Directory:** `docs/`

- [ ] Create user guide: `docs/3D_FKK_User_Guide.md`
  - Getting started with 3D FKK filtering
  - Filter type selection guide
  - Parameter tuning tips
  - Troubleshooting common issues

- [ ] Create workflow examples: `docs/3D_FKK_Workflows.md`
  - Ground roll removal workflow
  - Linear noise attenuation workflow
  - Coherent noise removal workflow
  - Quality control checklist

### 8.2 Developer Documentation
**Directory:** `docs/`

- [ ] Create architecture document: `docs/3D_FKK_Architecture.md`
  - Module overview
  - Data flow diagrams
  - GPU computing patterns
  - Extension points

- [ ] Create API reference: `docs/3D_FKK_API.md`
  - Class documentation
  - Method signatures
  - Usage examples

### 8.3 Performance Tuning Guide
**File:** `docs/3D_FKK_Performance.md`

- [ ] Document performance considerations
  - GPU memory requirements by volume size
  - Chunk size recommendations
  - Multi-GPU configuration
  - CPU fallback scenarios

### 8.4 Final Polish

- [ ] Add logging throughout
  - Processing progress
  - Memory usage
  - GPU utilization
  - Error diagnostics

- [ ] Add configuration persistence
  - Remember last-used settings
  - Workspace presets
  - Export/import settings

- [ ] Performance optimization pass
  - Profile hotspots
  - Optimize GPU kernel launches
  - Reduce CPU-GPU transfers

- [ ] Error handling improvements
  - Graceful degradation
  - User-friendly error messages
  - Recovery suggestions

---

## Dependencies

### New Python Packages
```
cupy-cuda12x>=12.0  # GPU FFT (or cupy-cuda11x for CUDA 11)
# PyTorch already in project for GPU operations
```

### Existing (already in project)
```
numpy
scipy
pyqt6
pyqtgraph
zarr
```

---

## Estimated Complexity

| Phase | Components | Complexity | Dependencies |
|-------|------------|------------|--------------|
| 1 | Data Models | Medium | None |
| 2 | GPU Compute | High | Phase 1 |
| 3 | Memory Mgmt | High | Phase 1, 2 |
| 4 | Parallel | Medium | Phase 2, 3 |
| 5 | UI | High | Phase 1, 2 |
| 6 | Integration | Medium | Phase 1-5 |
| 7 | Testing | Medium | Phase 1-6 |
| 8 | Docs | Low | Phase 1-7 |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| GPU memory limits | Chunked processing with fallback |
| FFT accuracy | Validate against scipy reference |
| UI responsiveness | Background threads, progress feedback |
| Large file I/O | Zarr chunking, memory mapping |
| Multi-GPU complexity | Start single-GPU, add multi-GPU later |

---

## Success Criteria

1. **Correctness:** FKK filter accurately attenuates target velocities
2. **Performance:** 512³ volume processed in < 500ms on modern GPU
3. **Memory:** Can process volumes 10x larger than GPU memory
4. **Usability:** Interactive filter design with real-time preview
5. **Integration:** Seamless workflow with existing 2D FK tools
6. **Reliability:** Graceful handling of edge cases and errors
