# PSTM UI Integration Plan for SeisProc

## Overview

This document outlines a dual-phased approach to integrate the Kirchhoff Pre-Stack Time Migration (PSTM) functionality into the SeisProc PyQt6 application. The computational backend (Phases 4-6 of the implementation plan) is complete with 391+ passing tests. This plan focuses on UI integration.

**Document Version:** 1.0
**Created:** December 2024
**Status:** Ready for Implementation

---

## Architecture Context

### Current SeisProc Structure
```
SeisProc Application
├── main.py                    # Entry point
├── main_window.py             # Main controller (MVC)
├── views/
│   ├── control_panel.py       # Algorithm parameters
│   ├── seismic_viewer_pyqtgraph.py
│   ├── gather_navigation_panel.py
│   ├── segy_import_dialog.py
│   ├── fk_designer_dialog.py
│   └── fkk_designer_dialog.py # Pattern for PSTM wizard
├── processors/
│   ├── base_processor.py      # Processor registry
│   ├── migration/             # PSTM backend (COMPLETE)
│   │   ├── kirchhoff_migrator.py
│   │   ├── traveltime*.py
│   │   ├── orchestrator.py
│   │   ├── checkpoint.py
│   │   └── ...
├── models/
│   ├── velocity_model.py      # Velocity structures
│   ├── migration_config.py    # Job configuration
│   ├── migration_job.py       # Job management
│   └── binning.py             # Offset/azimuth binning
└── seisio/
    ├── gather_readers.py      # Data access
    └── migration_output.py    # Output writing
```

### Integration Points
1. **Control Panel** - Add PSTM algorithm option
2. **Main Window** - Handle PSTM signals, manage dialogs
3. **New Dialogs** - Wizard, monitor, velocity editor
4. **Menu System** - Migration submenu

---

## Phase 1: Quick Integration (Demo Capability)

**Goal:** Enable basic PSTM processing on current gather for immediate demo capability
**Timeline:** 3-5 days
**Outcome:** Users can apply PSTM to single gathers with basic parameters

### Task 1.1: Register PSTM Processor
**File:** `processors/__init__.py`
**Effort:** 0.5 hours

- [ ] Import KirchhoffMigrator from migration module
- [ ] Add to PROCESSOR_REGISTRY dictionary
- [ ] Verify import works without errors

```python
# Add to processors/__init__.py
from .migration.kirchhoff_migrator import KirchhoffMigrator
PROCESSOR_REGISTRY['KirchhoffPSTM'] = KirchhoffMigrator
```

### Task 1.2: Add PSTM to Algorithm Dropdown
**File:** `views/control_panel.py`
**Effort:** 1 hour

- [ ] Add "Kirchhoff PSTM" to algorithm_combo items
- [ ] Update `_on_algorithm_changed()` to handle new index
- [ ] Create placeholder for PSTM parameter group

```python
# In _create_algorithm_selector()
self.algorithm_combo.addItems([
    "Bandpass Filter",
    "TF-Denoise (S-Transform)",
    "FK Filter",
    "3D FKK Filter",
    "Kirchhoff PSTM"  # NEW - index 4
])
```

### Task 1.3: Create PSTM Parameter Group
**File:** `views/control_panel.py`
**Effort:** 3-4 hours

- [ ] Create `_create_pstm_group()` method
- [ ] Add velocity input (QDoubleSpinBox, range 1000-8000 m/s)
- [ ] Add aperture input (QDoubleSpinBox, range 100-10000 m)
- [ ] Add max angle input (QDoubleSpinBox, range 10-80 degrees)
- [ ] Add "Apply PSTM" button
- [ ] Add "Open Wizard..." button (placeholder for Phase 2)
- [ ] Add GPU checkbox (if GPU available)
- [ ] Connect to `_on_algorithm_changed()` show/hide logic

```python
def _create_pstm_group(self) -> QGroupBox:
    """Create PSTM parameter controls."""
    group = QGroupBox("Kirchhoff PSTM")
    layout = QVBoxLayout()

    # Velocity
    vel_layout = QHBoxLayout()
    vel_layout.addWidget(QLabel("Velocity (m/s):"))
    self.pstm_velocity_spin = QDoubleSpinBox()
    self.pstm_velocity_spin.setRange(1000, 8000)
    self.pstm_velocity_spin.setValue(2500)
    self.pstm_velocity_spin.setSingleStep(100)
    vel_layout.addWidget(self.pstm_velocity_spin)
    layout.addLayout(vel_layout)

    # Aperture
    aperture_layout = QHBoxLayout()
    aperture_layout.addWidget(QLabel("Aperture (m):"))
    self.pstm_aperture_spin = QDoubleSpinBox()
    self.pstm_aperture_spin.setRange(100, 10000)
    self.pstm_aperture_spin.setValue(3000)
    self.pstm_aperture_spin.setSingleStep(100)
    aperture_layout.addWidget(self.pstm_aperture_spin)
    layout.addLayout(aperture_layout)

    # Max angle
    angle_layout = QHBoxLayout()
    angle_layout.addWidget(QLabel("Max Angle (deg):"))
    self.pstm_angle_spin = QDoubleSpinBox()
    self.pstm_angle_spin.setRange(10, 80)
    self.pstm_angle_spin.setValue(60)
    angle_layout.addWidget(self.pstm_angle_spin)
    layout.addLayout(angle_layout)

    # Apply button
    self.pstm_apply_btn = QPushButton("Apply PSTM")
    self.pstm_apply_btn.clicked.connect(self._on_pstm_apply)
    layout.addWidget(self.pstm_apply_btn)

    # Wizard button
    self.pstm_wizard_btn = QPushButton("Open PSTM Wizard...")
    self.pstm_wizard_btn.clicked.connect(lambda: self.pstm_wizard_requested.emit())
    layout.addWidget(self.pstm_wizard_btn)

    group.setLayout(layout)
    return group
```

### Task 1.4: Add PSTM Signal
**File:** `views/control_panel.py`
**Effort:** 0.5 hours

- [ ] Add `pstm_apply_requested` signal (emits velocity, aperture, angle)
- [ ] Add `pstm_wizard_requested` signal (no parameters)
- [ ] Implement `_on_pstm_apply()` method to emit signal

```python
# Add to class signals
pstm_apply_requested = pyqtSignal(float, float, float)  # velocity, aperture, angle
pstm_wizard_requested = pyqtSignal()

def _on_pstm_apply(self):
    """Emit PSTM apply request with current parameters."""
    self.pstm_apply_requested.emit(
        self.pstm_velocity_spin.value(),
        self.pstm_aperture_spin.value(),
        self.pstm_angle_spin.value()
    )
```

### Task 1.5: Handle PSTM in Main Window
**File:** `main_window.py`
**Effort:** 4-6 hours

- [ ] Import migration components
- [ ] Connect control panel signals to handlers
- [ ] Implement `_on_pstm_apply_requested()` method
- [ ] Create gather from current data with geometry
- [ ] Instantiate KirchhoffMigrator with parameters
- [ ] Execute migration on current gather
- [ ] Display result in processed viewer
- [ ] Handle errors with QMessageBox
- [ ] Show progress in status bar

```python
# In __init__, connect signals
self.control_panel.pstm_apply_requested.connect(self._on_pstm_apply_requested)
self.control_panel.pstm_wizard_requested.connect(self._on_pstm_wizard_requested)

def _on_pstm_apply_requested(self, velocity: float, aperture: float, max_angle: float):
    """Handle PSTM apply request from control panel."""
    if self.input_data is None:
        QMessageBox.warning(self, "No Data", "Please load seismic data first.")
        return

    try:
        self.statusBar().showMessage("Running PSTM migration...")
        QApplication.processEvents()

        # Create velocity model
        from models.velocity_model import create_constant_velocity
        v_model = create_constant_velocity(velocity)

        # Create migration config
        from models.migration_config import MigrationConfig, OutputGrid

        # Build output grid from input data dimensions
        output_grid = OutputGrid(
            n_time=self.input_data.n_samples,
            n_inline=1,
            n_xline=self.input_data.n_traces,
            dt=self.input_data.sample_rate / 1000.0,
        )

        config = MigrationConfig(
            output_grid=output_grid,
            max_aperture_m=aperture,
            max_angle_deg=max_angle,
        )

        # Create migrator and run
        from processors.migration import KirchhoffMigrator
        migrator = KirchhoffMigrator(v_model, config)

        # Create gather with geometry (use offsets from headers if available)
        gather = self._create_gather_for_migration()

        result = migrator.migrate_gather(gather)

        # Display result
        from models.seismic_data import SeismicData
        migrated_data = SeismicData(
            traces=result.migrated_volume.squeeze(),
            sample_rate=self.input_data.sample_rate,
            metadata={'processor': 'KirchhoffPSTM', 'velocity': velocity}
        )

        self.processed_data = migrated_data
        self.processed_viewer.set_data(migrated_data)

        # Compute difference
        self._update_difference_view()

        self.statusBar().showMessage(f"PSTM complete (v={velocity} m/s)", 5000)

    except Exception as e:
        logger.error(f"PSTM failed: {e}", exc_info=True)
        QMessageBox.critical(self, "PSTM Error", f"Migration failed:\n{e}")
        self.statusBar().showMessage("PSTM failed")
```

### Task 1.6: Create Gather with Geometry Helper
**File:** `main_window.py`
**Effort:** 2 hours

- [ ] Implement `_create_gather_for_migration()` method
- [ ] Extract offsets from headers if available
- [ ] Generate synthetic offsets if not available
- [ ] Extract source/receiver coordinates if available
- [ ] Return Gather object compatible with migrator

```python
def _create_gather_for_migration(self):
    """Create Gather object from current data with geometry."""
    from seisio.gather_readers import Gather
    import numpy as np

    n_traces = self.input_data.n_traces

    # Try to get offsets from headers
    offsets = None
    if self.input_data.headers and 'offset' in self.input_data.headers:
        offsets = np.array(self.input_data.headers['offset'], dtype=np.float32)
    else:
        # Generate synthetic offsets (assume regular spacing)
        offsets = np.linspace(100, 3000, n_traces, dtype=np.float32)

    # Azimuths (default to 0 if not available)
    azimuths = np.zeros(n_traces, dtype=np.float32)
    if self.input_data.headers and 'azimuth' in self.input_data.headers:
        azimuths = np.array(self.input_data.headers['azimuth'], dtype=np.float32)

    return Gather(
        gather_id="current",
        trace_numbers=np.arange(n_traces, dtype=np.int32),
        data=self.input_data.traces.T,  # Migrator expects (n_traces, n_samples)
        offsets=offsets,
        azimuths=azimuths,
    )
```

### Task 1.7: Add Migration Menu Item
**File:** `main_window.py`
**Effort:** 1 hour

- [ ] Add "Migration" submenu to Process menu
- [ ] Add "Kirchhoff PSTM Wizard..." action (Ctrl+M)
- [ ] Connect to placeholder handler

```python
# In _create_menu_bar(), after existing process_menu items
process_menu.addSeparator()

# Migration submenu
migration_menu = process_menu.addMenu("&Migration")

pstm_wizard_action = QAction("&Kirchhoff PSTM Wizard...", self)
pstm_wizard_action.setShortcut("Ctrl+M")
pstm_wizard_action.setToolTip("Open the PSTM configuration wizard")
pstm_wizard_action.triggered.connect(self._on_pstm_wizard_requested)
migration_menu.addAction(pstm_wizard_action)
```

### Task 1.8: Testing and Validation
**Effort:** 4-6 hours

- [ ] Test with synthetic data (flat reflector)
- [ ] Test with real gather data
- [ ] Verify parameter changes affect output
- [ ] Test error handling (no data loaded)
- [ ] Test with/without header geometry
- [ ] Performance check on typical gather size
- [ ] Verify GPU acceleration works if available

### Phase 1 Deliverables Checklist
- [ ] PSTM appears in algorithm dropdown
- [ ] Parameter group shows/hides correctly
- [ ] Apply button runs migration on current gather
- [ ] Result displays in processed viewer
- [ ] Difference view updates
- [ ] Status bar shows progress/completion
- [ ] Errors show informative messages
- [ ] Menu item exists (placeholder for Phase 2)

---

## Phase 2: Production Wizard (Full Workflow)

**Goal:** Complete production-ready PSTM workflow with wizard, monitoring, and job management
**Timeline:** 2-3 weeks
**Outcome:** Users can configure and run full dataset migrations with checkpointing

### Task 2.1: Create PSTM Wizard Dialog Structure
**File:** `views/pstm_wizard_dialog.py` (NEW)
**Effort:** 8-12 hours

- [ ] Create QWizard-based dialog class
- [ ] Define 7 wizard pages
- [ ] Implement page navigation logic
- [ ] Add validation between pages
- [ ] Store configuration state across pages

```python
"""
PSTM Configuration Wizard

Multi-page wizard for configuring Kirchhoff PSTM jobs.
"""
from PyQt6.QtWidgets import (
    QWizard, QWizardPage, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog,
    QDoubleSpinBox, QSpinBox, QComboBox, QGroupBox,
    QTableWidget, QTextEdit, QProgressBar
)
from PyQt6.QtCore import pyqtSignal
from pathlib import Path

from models.migration_job import MigrationJobConfig, JOB_TEMPLATES


class PSTMWizard(QWizard):
    """Multi-page PSTM configuration wizard."""

    job_configured = pyqtSignal(object)  # Emits MigrationJobConfig

    def __init__(self, parent=None, initial_file: str = None):
        super().__init__(parent)
        self.setWindowTitle("Kirchhoff PSTM Wizard")
        self.resize(800, 600)

        self.initial_file = initial_file
        self._config = MigrationJobConfig(
            name="New Migration Job",
            input_file=initial_file or "",
            output_directory=""
        )

        # Add pages
        self.addPage(JobSetupPage(self))
        self.addPage(VelocityModelPage(self))
        self.addPage(HeaderMappingPage(self))
        self.addPage(OutputGridPage(self))
        self.addPage(BinningConfigPage(self))
        self.addPage(AdvancedOptionsPage(self))
        self.addPage(ReviewLaunchPage(self))

        self.finished.connect(self._on_finished)

    def _on_finished(self, result):
        if result == QWizard.DialogCode.Accepted:
            self.job_configured.emit(self._config)
```

### Task 2.2: Implement Job Setup Page
**File:** `views/pstm_wizard_dialog.py`
**Effort:** 3-4 hours

- [ ] Job name input field
- [ ] Input file selector with browse button
- [ ] Output directory selector
- [ ] Template dropdown (land_3d, marine, etc.)
- [ ] File preview panel (trace count, sample rate, duration)
- [ ] Validation: files exist, writable directory

```python
class JobSetupPage(QWizardPage):
    """Page 1: Basic job configuration."""

    def __init__(self, wizard):
        super().__init__(wizard)
        self.wizard = wizard
        self.setTitle("Job Setup")
        self.setSubTitle("Configure basic job parameters and file paths")

        layout = QVBoxLayout(self)

        # Job name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Job Name:"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Enter descriptive job name")
        name_layout.addWidget(self.name_edit)
        layout.addLayout(name_layout)

        # Template selection
        template_layout = QHBoxLayout()
        template_layout.addWidget(QLabel("Template:"))
        self.template_combo = QComboBox()
        self.template_combo.addItems(["Custom", "Land 3D", "Marine", "Wide Azimuth OVT", "Full Stack"])
        self.template_combo.currentIndexChanged.connect(self._on_template_changed)
        template_layout.addWidget(self.template_combo)
        layout.addLayout(template_layout)

        # Input file
        input_group = QGroupBox("Input Data")
        input_layout = QVBoxLayout(input_group)

        file_layout = QHBoxLayout()
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Select input SEG-Y file")
        file_layout.addWidget(self.input_edit)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_input)
        file_layout.addWidget(browse_btn)
        input_layout.addLayout(file_layout)

        # File info preview
        self.file_info_label = QLabel("No file selected")
        input_layout.addWidget(self.file_info_label)

        layout.addWidget(input_group)

        # Output directory
        output_group = QGroupBox("Output Location")
        output_layout = QHBoxLayout(output_group)
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Select output directory")
        output_layout.addWidget(self.output_edit)
        output_browse_btn = QPushButton("Browse...")
        output_browse_btn.clicked.connect(self._browse_output)
        output_layout.addWidget(output_browse_btn)
        layout.addWidget(output_group)

        # Register fields for validation
        self.registerField("jobName*", self.name_edit)
        self.registerField("inputFile*", self.input_edit)
        self.registerField("outputDir*", self.output_edit)
```

### Task 2.3: Implement Velocity Model Page
**File:** `views/pstm_wizard_dialog.py`
**Effort:** 4-6 hours

- [ ] Constant velocity input
- [ ] Velocity gradient input
- [ ] Load from file option (v(z), v(x,z))
- [ ] Velocity preview plot (1D profile or 2D cross-section)
- [ ] Validation: positive velocities, reasonable range

```python
class VelocityModelPage(QWizardPage):
    """Page 2: Velocity model configuration."""

    def __init__(self, wizard):
        super().__init__(wizard)
        self.wizard = wizard
        self.setTitle("Velocity Model")
        self.setSubTitle("Define the velocity model for migration")

        layout = QVBoxLayout(self)

        # Velocity type selection
        type_group = QGroupBox("Velocity Type")
        type_layout = QVBoxLayout(type_group)

        self.const_radio = QRadioButton("Constant Velocity")
        self.const_radio.setChecked(True)
        self.gradient_radio = QRadioButton("Linear Gradient v(z) = v0 + k*z")
        self.file_radio = QRadioButton("Load from File")

        type_layout.addWidget(self.const_radio)
        type_layout.addWidget(self.gradient_radio)
        type_layout.addWidget(self.file_radio)
        layout.addWidget(type_group)

        # Parameters group
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout(params_group)

        # V0
        v0_layout = QHBoxLayout()
        v0_layout.addWidget(QLabel("V0 (m/s):"))
        self.v0_spin = QDoubleSpinBox()
        self.v0_spin.setRange(500, 8000)
        self.v0_spin.setValue(2500)
        self.v0_spin.setSingleStep(100)
        v0_layout.addWidget(self.v0_spin)
        params_layout.addLayout(v0_layout)

        # Gradient
        grad_layout = QHBoxLayout()
        grad_layout.addWidget(QLabel("Gradient (1/s):"))
        self.gradient_spin = QDoubleSpinBox()
        self.gradient_spin.setRange(-2.0, 2.0)
        self.gradient_spin.setValue(0.0)
        self.gradient_spin.setSingleStep(0.1)
        self.gradient_spin.setDecimals(3)
        grad_layout.addWidget(self.gradient_spin)
        params_layout.addLayout(grad_layout)

        # File path
        file_layout = QHBoxLayout()
        self.vel_file_edit = QLineEdit()
        self.vel_file_edit.setEnabled(False)
        file_layout.addWidget(self.vel_file_edit)
        self.vel_browse_btn = QPushButton("Browse...")
        self.vel_browse_btn.setEnabled(False)
        file_layout.addWidget(self.vel_browse_btn)
        params_layout.addLayout(file_layout)

        layout.addWidget(params_group)

        # Preview plot placeholder
        preview_group = QGroupBox("Velocity Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.vel_preview_label = QLabel("Constant velocity: 2500 m/s")
        preview_layout.addWidget(self.vel_preview_label)
        layout.addWidget(preview_group)

        # Connect radio buttons
        self.const_radio.toggled.connect(self._update_ui_state)
        self.gradient_radio.toggled.connect(self._update_ui_state)
        self.file_radio.toggled.connect(self._update_ui_state)
```

### Task 2.4: Implement Header Mapping Page
**File:** `views/pstm_wizard_dialog.py`
**Effort:** 6-8 hours

- [ ] Required headers table (SX, SY, GX, GY, OFFSET, etc.)
- [ ] Available headers dropdown per row
- [ ] Auto-detect button (match by name patterns)
- [ ] Validation status per header
- [ ] Sample values preview
- [ ] Warning for unmapped required headers

```python
class HeaderMappingPage(QWizardPage):
    """Page 3: Map input headers to required fields."""

    REQUIRED_HEADERS = [
        ('source_x', 'Source X', True),
        ('source_y', 'Source Y', True),
        ('receiver_x', 'Receiver X', True),
        ('receiver_y', 'Receiver Y', True),
        ('offset', 'Offset', False),  # Can compute
        ('azimuth', 'Azimuth', False),  # Can compute
        ('inline', 'Inline', False),
        ('xline', 'Crossline', False),
    ]

    def __init__(self, wizard):
        super().__init__(wizard)
        self.wizard = wizard
        self.setTitle("Header Mapping")
        self.setSubTitle("Map SEG-Y trace headers to migration parameters")

        layout = QVBoxLayout(self)

        # Auto-detect button
        auto_layout = QHBoxLayout()
        auto_btn = QPushButton("Auto-Detect Headers")
        auto_btn.clicked.connect(self._auto_detect)
        auto_layout.addWidget(auto_btn)
        auto_layout.addStretch()
        layout.addLayout(auto_layout)

        # Mapping table
        self.mapping_table = QTableWidget()
        self.mapping_table.setColumnCount(4)
        self.mapping_table.setHorizontalHeaderLabels([
            "Required Field", "Input Header", "Status", "Sample Value"
        ])
        self.mapping_table.setRowCount(len(self.REQUIRED_HEADERS))

        for i, (field, label, required) in enumerate(self.REQUIRED_HEADERS):
            # Field name
            name_item = QTableWidgetItem(label)
            if required:
                name_item.setText(f"{label} *")
            self.mapping_table.setItem(i, 0, name_item)

            # Header dropdown
            combo = QComboBox()
            combo.addItem("-- Not Mapped --")
            self.mapping_table.setCellWidget(i, 1, combo)

            # Status
            status_item = QTableWidgetItem("Not mapped")
            self.mapping_table.setItem(i, 2, status_item)

            # Sample value
            sample_item = QTableWidgetItem("-")
            self.mapping_table.setItem(i, 3, sample_item)

        layout.addWidget(self.mapping_table)

        # Validation summary
        self.validation_label = QLabel("Map required headers (*) to continue")
        layout.addWidget(self.validation_label)
```

### Task 2.5: Implement Output Grid Page
**File:** `views/pstm_wizard_dialog.py`
**Effort:** 4-5 hours

- [ ] Time range inputs (min, max, sample rate)
- [ ] Inline range inputs (min, max, step)
- [ ] Crossline range inputs (min, max, step)
- [ ] Origin coordinates (X, Y)
- [ ] Grid spacing (inline, crossline)
- [ ] Grid size preview (dimensions, memory estimate)
- [ ] Auto-detect from headers button

```python
class OutputGridPage(QWizardPage):
    """Page 4: Define output grid geometry."""

    def __init__(self, wizard):
        super().__init__(wizard)
        self.wizard = wizard
        self.setTitle("Output Grid")
        self.setSubTitle("Define the output image grid dimensions")

        layout = QVBoxLayout(self)

        # Time parameters
        time_group = QGroupBox("Time Axis")
        time_layout = QGridLayout(time_group)

        time_layout.addWidget(QLabel("Min Time (ms):"), 0, 0)
        self.time_min_spin = QDoubleSpinBox()
        self.time_min_spin.setRange(0, 20000)
        self.time_min_spin.setValue(0)
        time_layout.addWidget(self.time_min_spin, 0, 1)

        time_layout.addWidget(QLabel("Max Time (ms):"), 0, 2)
        self.time_max_spin = QDoubleSpinBox()
        self.time_max_spin.setRange(100, 20000)
        self.time_max_spin.setValue(4000)
        time_layout.addWidget(self.time_max_spin, 0, 3)

        time_layout.addWidget(QLabel("Sample Rate (ms):"), 1, 0)
        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setRange(0.5, 16)
        self.dt_spin.setValue(4)
        self.dt_spin.setDecimals(1)
        time_layout.addWidget(self.dt_spin, 1, 1)

        layout.addWidget(time_group)

        # Spatial parameters
        spatial_group = QGroupBox("Spatial Grid")
        spatial_layout = QGridLayout(spatial_group)

        # Inline
        spatial_layout.addWidget(QLabel("Inline Min:"), 0, 0)
        self.il_min_spin = QSpinBox()
        self.il_min_spin.setRange(1, 100000)
        self.il_min_spin.setValue(1)
        spatial_layout.addWidget(self.il_min_spin, 0, 1)

        spatial_layout.addWidget(QLabel("Inline Max:"), 0, 2)
        self.il_max_spin = QSpinBox()
        self.il_max_spin.setRange(1, 100000)
        self.il_max_spin.setValue(100)
        spatial_layout.addWidget(self.il_max_spin, 0, 3)

        spatial_layout.addWidget(QLabel("Inline Step:"), 0, 4)
        self.il_step_spin = QSpinBox()
        self.il_step_spin.setRange(1, 100)
        self.il_step_spin.setValue(1)
        spatial_layout.addWidget(self.il_step_spin, 0, 5)

        # Similar for crossline...

        layout.addWidget(spatial_group)

        # Grid preview
        preview_group = QGroupBox("Grid Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.grid_preview_label = QLabel("Output: 1001 samples x 100 inlines x 100 xlines\nMemory: ~38 MB per volume")
        preview_layout.addWidget(self.grid_preview_label)
        layout.addWidget(preview_group)
```

### Task 2.6: Implement Binning Configuration Page
**File:** `views/pstm_wizard_dialog.py`
**Effort:** 5-6 hours

- [ ] Preset dropdown (land_3d, marine, ovt, full_stack)
- [ ] Custom bin table editor
- [ ] Add/remove bin buttons
- [ ] Offset/azimuth range per bin
- [ ] Bin coverage preview
- [ ] Validation: no overlapping bins (optional warning)

```python
class BinningConfigPage(QWizardPage):
    """Page 5: Configure offset/azimuth binning."""

    def __init__(self, wizard):
        super().__init__(wizard)
        self.wizard = wizard
        self.setTitle("Binning Configuration")
        self.setSubTitle("Define offset and azimuth bins for migration")

        layout = QVBoxLayout(self)

        # Preset selection
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Custom",
            "Land 3D (10 offset bins)",
            "Marine (6 offset bins)",
            "Wide Azimuth OVT (16 bins)",
            "Full Stack (1 bin)"
        ])
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        preset_layout.addWidget(self.preset_combo)
        layout.addLayout(preset_layout)

        # Bin table
        self.bin_table = QTableWidget()
        self.bin_table.setColumnCount(6)
        self.bin_table.setHorizontalHeaderLabels([
            "Name", "Offset Min", "Offset Max", "Az Min", "Az Max", "Enabled"
        ])
        layout.addWidget(self.bin_table)

        # Add/remove buttons
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Bin")
        add_btn.clicked.connect(self._add_bin)
        btn_layout.addWidget(add_btn)
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._remove_bin)
        btn_layout.addWidget(remove_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Summary
        self.bin_summary_label = QLabel("10 bins configured")
        layout.addWidget(self.bin_summary_label)
```

### Task 2.7: Implement Advanced Options Page
**File:** `views/pstm_wizard_dialog.py`
**Effort:** 3-4 hours

- [ ] Traveltime mode dropdown (straight, curved, VTI)
- [ ] Aperture control parameters
- [ ] Antialiasing checkbox and parameters
- [ ] Checkpointing enable/disable
- [ ] Memory limit setting
- [ ] GPU preference checkbox
- [ ] Parallel workers count

```python
class AdvancedOptionsPage(QWizardPage):
    """Page 6: Advanced migration options."""

    def __init__(self, wizard):
        super().__init__(wizard)
        self.wizard = wizard
        self.setTitle("Advanced Options")
        self.setSubTitle("Configure advanced migration parameters")

        layout = QVBoxLayout(self)

        # Traveltime group
        tt_group = QGroupBox("Traveltime Calculation")
        tt_layout = QVBoxLayout(tt_group)

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.tt_mode_combo = QComboBox()
        self.tt_mode_combo.addItems(["Straight Ray", "Curved Ray", "VTI Anisotropic"])
        mode_layout.addWidget(self.tt_mode_combo)
        tt_layout.addLayout(mode_layout)

        layout.addWidget(tt_group)

        # Aperture group
        aperture_group = QGroupBox("Aperture Control")
        aperture_layout = QGridLayout(aperture_group)

        aperture_layout.addWidget(QLabel("Max Aperture (m):"), 0, 0)
        self.aperture_spin = QDoubleSpinBox()
        self.aperture_spin.setRange(100, 20000)
        self.aperture_spin.setValue(3000)
        aperture_layout.addWidget(self.aperture_spin, 0, 1)

        aperture_layout.addWidget(QLabel("Max Angle (deg):"), 1, 0)
        self.angle_spin = QDoubleSpinBox()
        self.angle_spin.setRange(10, 85)
        self.angle_spin.setValue(60)
        aperture_layout.addWidget(self.angle_spin, 1, 1)

        layout.addWidget(aperture_group)

        # Processing group
        proc_group = QGroupBox("Processing Options")
        proc_layout = QVBoxLayout(proc_group)

        self.antialias_check = QCheckBox("Enable Antialiasing")
        self.antialias_check.setChecked(True)
        proc_layout.addWidget(self.antialias_check)

        self.checkpoint_check = QCheckBox("Enable Checkpointing")
        self.checkpoint_check.setChecked(True)
        proc_layout.addWidget(self.checkpoint_check)

        self.gpu_check = QCheckBox("Use GPU Acceleration")
        self.gpu_check.setChecked(True)
        proc_layout.addWidget(self.gpu_check)

        workers_layout = QHBoxLayout()
        workers_layout.addWidget(QLabel("Parallel Workers:"))
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 16)
        self.workers_spin.setValue(4)
        workers_layout.addWidget(self.workers_spin)
        proc_layout.addLayout(workers_layout)

        layout.addWidget(proc_group)
```

### Task 2.8: Implement Review & Launch Page
**File:** `views/pstm_wizard_dialog.py`
**Effort:** 3-4 hours

- [ ] Configuration summary text
- [ ] Resource estimate (time, memory, disk)
- [ ] Validation status summary
- [ ] Save configuration button
- [ ] Run now / Run later options

```python
class ReviewLaunchPage(QWizardPage):
    """Page 7: Review configuration and launch job."""

    def __init__(self, wizard):
        super().__init__(wizard)
        self.wizard = wizard
        self.setTitle("Review & Launch")
        self.setSubTitle("Review configuration and start migration")

        layout = QVBoxLayout(self)

        # Configuration summary
        summary_group = QGroupBox("Configuration Summary")
        summary_layout = QVBoxLayout(summary_group)
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        summary_layout.addWidget(self.summary_text)
        layout.addWidget(summary_group)

        # Resource estimate
        resource_group = QGroupBox("Resource Estimate")
        resource_layout = QVBoxLayout(resource_group)
        self.resource_label = QLabel(
            "Estimated time: ~45 minutes\n"
            "Memory required: ~4 GB\n"
            "Output size: ~2.5 GB"
        )
        resource_layout.addWidget(self.resource_label)
        layout.addWidget(resource_group)

        # Validation status
        validation_group = QGroupBox("Validation")
        validation_layout = QVBoxLayout(validation_group)
        self.validation_text = QTextEdit()
        self.validation_text.setReadOnly(True)
        self.validation_text.setMaximumHeight(100)
        validation_layout.addWidget(self.validation_text)
        layout.addWidget(validation_group)

        # Action buttons
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Save Configuration...")
        save_btn.clicked.connect(self._save_config)
        btn_layout.addWidget(save_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def initializePage(self):
        """Called when page is shown - update summary."""
        self._update_summary()
        self._update_validation()
        self._update_resource_estimate()
```

### Task 2.9: Create Migration Monitor Dialog
**File:** `views/migration_monitor_dialog.py` (NEW)
**Effort:** 8-10 hours

- [ ] Job progress bar (overall)
- [ ] Current bin progress bar
- [ ] Elapsed time / estimated remaining
- [ ] Status log (scrolling text)
- [ ] Pause/Resume/Cancel buttons
- [ ] View output button (enabled when complete)
- [ ] Background thread for job execution
- [ ] Signal handling for progress updates

```python
"""
Migration Job Monitor Dialog

Displays progress and allows control of running migration jobs.
"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QTextEdit, QGroupBox, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer
import time
import logging

from models.migration_job import MigrationJobConfig
from processors.migration.orchestrator import MigrationOrchestrator, MigrationProgress
from processors.migration.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


class MigrationWorker(QThread):
    """Background worker for migration execution."""

    progress_updated = pyqtSignal(object)  # MigrationProgress
    log_message = pyqtSignal(str)
    finished = pyqtSignal(bool, str)  # success, message

    def __init__(self, job_config: MigrationJobConfig):
        super().__init__()
        self.job_config = job_config
        self._cancelled = False
        self._paused = False

    def run(self):
        """Execute migration job."""
        try:
            # Create orchestrator
            orchestrator = MigrationOrchestrator(
                job_config=self.job_config,
                # ... other params
                progress_callback=self._on_progress
            )

            orchestrator.setup()
            result = orchestrator.run()

            if self._cancelled:
                self.finished.emit(False, "Job cancelled by user")
            else:
                self.finished.emit(True, "Migration completed successfully")

        except Exception as e:
            logger.error(f"Migration failed: {e}", exc_info=True)
            self.finished.emit(False, str(e))

    def _on_progress(self, progress: MigrationProgress):
        self.progress_updated.emit(progress)

    def cancel(self):
        self._cancelled = True

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False


class MigrationMonitorDialog(QDialog):
    """Dialog for monitoring migration job progress."""

    def __init__(self, job_config: MigrationJobConfig, parent=None):
        super().__init__(parent)
        self.job_config = job_config
        self.worker = None

        self.setWindowTitle(f"Migration: {job_config.name}")
        self.resize(600, 500)
        self.setModal(False)  # Allow interaction with main window

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Job info
        info_group = QGroupBox("Job Information")
        info_layout = QVBoxLayout(info_group)
        self.job_label = QLabel(f"Job: {self.job_config.name}")
        info_layout.addWidget(self.job_label)
        layout.addWidget(info_group)

        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)

        # Overall progress
        overall_layout = QHBoxLayout()
        overall_layout.addWidget(QLabel("Overall:"))
        self.overall_progress = QProgressBar()
        overall_layout.addWidget(self.overall_progress)
        self.overall_label = QLabel("0%")
        overall_layout.addWidget(self.overall_label)
        progress_layout.addLayout(overall_layout)

        # Current bin progress
        bin_layout = QHBoxLayout()
        bin_layout.addWidget(QLabel("Current Bin:"))
        self.bin_progress = QProgressBar()
        bin_layout.addWidget(self.bin_progress)
        self.bin_label = QLabel("-")
        bin_layout.addWidget(self.bin_label)
        progress_layout.addLayout(bin_layout)

        # Time
        time_layout = QHBoxLayout()
        self.time_label = QLabel("Elapsed: 00:00:00 | Remaining: --:--:--")
        time_layout.addWidget(self.time_label)
        progress_layout.addLayout(time_layout)

        layout.addWidget(progress_group)

        # Log
        log_group = QGroupBox("Status Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_group)

        # Buttons
        btn_layout = QHBoxLayout()
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self._on_pause)
        btn_layout.addWidget(self.pause_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel)
        btn_layout.addWidget(self.cancel_btn)

        btn_layout.addStretch()

        self.view_btn = QPushButton("View Output")
        self.view_btn.setEnabled(False)
        self.view_btn.clicked.connect(self._on_view_output)
        btn_layout.addWidget(self.view_btn)

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        btn_layout.addWidget(self.close_btn)

        layout.addLayout(btn_layout)

        # Timer for elapsed time updates
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_time)
        self.start_time = None

    def start_job(self):
        """Start the migration job."""
        self.worker = MigrationWorker(self.job_config)
        self.worker.progress_updated.connect(self._on_progress)
        self.worker.log_message.connect(self._on_log)
        self.worker.finished.connect(self._on_finished)

        self.start_time = time.time()
        self.timer.start(1000)

        self._log("Starting migration job...")
        self.worker.start()
```

### Task 2.10: Integrate Wizard with Main Window
**File:** `main_window.py`
**Effort:** 3-4 hours

- [ ] Import wizard dialog
- [ ] Implement `_on_pstm_wizard_requested()` handler
- [ ] Handle wizard completion signal
- [ ] Launch monitor dialog for job execution
- [ ] Track running jobs
- [ ] Handle job completion/viewing results

```python
def _on_pstm_wizard_requested(self):
    """Open PSTM configuration wizard."""
    from views.pstm_wizard_dialog import PSTMWizard

    # Pass current file if loaded
    initial_file = None
    if hasattr(self, '_current_segy_path') and self._current_segy_path:
        initial_file = self._current_segy_path

    wizard = PSTMWizard(parent=self, initial_file=initial_file)
    wizard.job_configured.connect(self._on_pstm_job_configured)
    wizard.exec()

def _on_pstm_job_configured(self, job_config):
    """Handle completed PSTM wizard configuration."""
    from views.migration_monitor_dialog import MigrationMonitorDialog

    # Validate configuration
    errors = job_config.validate()
    if errors:
        QMessageBox.warning(
            self,
            "Configuration Errors",
            "Please fix the following errors:\n\n" + "\n".join(errors)
        )
        return

    # Ask to start now
    reply = QMessageBox.question(
        self,
        "Start Migration",
        f"Start migration job '{job_config.name}' now?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
    )

    if reply == QMessageBox.StandardButton.Yes:
        monitor = MigrationMonitorDialog(job_config, parent=self)
        monitor.show()
        monitor.start_job()

        # Track running job
        if not hasattr(self, '_migration_jobs'):
            self._migration_jobs = []
        self._migration_jobs.append(monitor)
```

### Task 2.11: Add Job Resume Functionality
**File:** `main_window.py`
**Effort:** 2-3 hours

- [ ] Implement `_resume_migration()` handler
- [ ] Show file dialog for checkpoint selection
- [ ] Validate checkpoint compatibility
- [ ] Launch monitor dialog for resumed job

```python
def _resume_migration(self):
    """Resume a migration job from checkpoint."""
    from PyQt6.QtWidgets import QFileDialog
    from processors.migration.checkpoint import find_resumable_jobs

    # Let user select output directory
    dir_path = QFileDialog.getExistingDirectory(
        self,
        "Select Migration Output Directory",
        "",
        QFileDialog.Option.ShowDirsOnly
    )

    if not dir_path:
        return

    # Find resumable jobs
    jobs = find_resumable_jobs(dir_path)

    if not jobs:
        QMessageBox.information(
            self,
            "No Resumable Jobs",
            "No incomplete migration jobs found in this directory."
        )
        return

    # If multiple jobs, let user select
    if len(jobs) > 1:
        # Show selection dialog
        pass
    else:
        job_info = jobs[0]

    # Resume the job
    # ... implementation
```

### Task 2.12: Testing and Documentation
**Effort:** 8-12 hours

- [ ] Unit tests for wizard pages
- [ ] Integration test: complete wizard flow
- [ ] Integration test: job execution and monitoring
- [ ] Integration test: checkpoint/resume
- [ ] Test with various input data sizes
- [ ] Test error handling throughout
- [ ] Update user documentation
- [ ] Add tooltips throughout wizard

### Phase 2 Deliverables Checklist
- [ ] PSTM Wizard opens from menu and control panel
- [ ] All 7 wizard pages functional
- [ ] Configuration saves/loads correctly
- [ ] Job launches from wizard
- [ ] Monitor dialog shows progress
- [ ] Pause/Resume/Cancel work
- [ ] Checkpointing functions correctly
- [ ] Completed jobs can be viewed
- [ ] Resume from checkpoint works
- [ ] Documentation updated

---

## File Summary

### New Files to Create
| File | Purpose | Phase |
|------|---------|-------|
| `views/pstm_wizard_dialog.py` | Multi-page configuration wizard | 2 |
| `views/migration_monitor_dialog.py` | Job progress monitoring | 2 |
| `views/velocity_editor_widget.py` | Velocity visualization (optional) | 2 |

### Files to Modify
| File | Changes | Phase |
|------|---------|-------|
| `processors/__init__.py` | Register KirchhoffMigrator | 1 |
| `views/control_panel.py` | Add PSTM algorithm and parameter group | 1 |
| `main_window.py` | Connect signals, handle PSTM, manage dialogs | 1, 2 |

### Existing Files to Leverage (No Changes)
- `processors/migration/*` - Complete PSTM backend
- `models/migration_job.py` - Job configuration
- `models/migration_config.py` - Migration parameters
- `models/velocity_model.py` - Velocity structures
- `models/binning.py` - Offset/azimuth binning
- `seisio/gather_readers.py` - Data access
- `seisio/migration_output.py` - Output writing

---

## Success Criteria

### Phase 1 Complete When:
1. User can select "Kirchhoff PSTM" from algorithm dropdown
2. User can set velocity, aperture, max angle
3. User can apply PSTM to current gather
4. Migrated result displays in processed viewer
5. Basic error handling works

### Phase 2 Complete When:
1. User can open wizard from menu (Ctrl+M)
2. User can configure complete job through wizard
3. User can launch job and see progress
4. User can pause/resume/cancel jobs
5. Checkpointing preserves progress
6. User can resume interrupted jobs
7. Completed output can be loaded and viewed

---

## Appendix: Signal Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Phase 1 Signal Flow                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ControlPanel                    MainWindow                      │
│  ┌──────────────┐               ┌──────────────┐                │
│  │ Apply PSTM   │───signal───▶  │ _on_pstm_    │                │
│  │ Button       │               │ apply_       │                │
│  └──────────────┘               │ requested()  │                │
│                                 └──────┬───────┘                │
│                                        │                         │
│                                        ▼                         │
│                                 ┌──────────────┐                │
│                                 │ Kirchhoff    │                │
│                                 │ Migrator     │                │
│                                 └──────┬───────┘                │
│                                        │                         │
│                                        ▼                         │
│  ProcessedViewer                ┌──────────────┐                │
│  ┌──────────────┐◀──set_data── │ Migrated     │                │
│  │ Display      │               │ Result       │                │
│  └──────────────┘               └──────────────┘                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        Phase 2 Signal Flow                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Menu/ControlPanel              MainWindow                       │
│  ┌──────────────┐               ┌──────────────┐                │
│  │ Open Wizard  │───signal───▶  │ _on_pstm_    │                │
│  │ Action       │               │ wizard_      │                │
│  └──────────────┘               │ requested()  │                │
│                                 └──────┬───────┘                │
│                                        │                         │
│                                        ▼                         │
│                                 ┌──────────────┐                │
│                                 │ PSTMWizard   │                │
│                                 │ Dialog       │                │
│                                 └──────┬───────┘                │
│                                        │ job_configured         │
│                                        ▼                         │
│                                 ┌──────────────┐                │
│                                 │ Migration    │                │
│                                 │ Monitor      │                │
│                                 │ Dialog       │                │
│                                 └──────┬───────┘                │
│                                        │                         │
│                                        ▼                         │
│                                 ┌──────────────┐                │
│  (Background Thread)            │ Migration    │                │
│  ┌──────────────┐◀─progress──  │ Worker       │                │
│  │ Orchestrator │               │ Thread       │                │
│  └──────────────┘               └──────────────┘                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```
