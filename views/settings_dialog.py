"""
Settings dialog for application configuration.

Provides UI for:
- Storage locations (Zarr/Parquet data directory)
- Dataset management options
- Display preferences
- Session behavior
"""
import logging
from pathlib import Path
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QLabel, QLineEdit, QPushButton, QFileDialog, QSpinBox,
    QCheckBox, QComboBox, QGroupBox, QFormLayout, QMessageBox,
    QDialogButtonBox, QSlider, QFrame
)
from PyQt6.QtCore import Qt
import multiprocessing

from models.app_settings import get_settings

# Try to import GPU detection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class SettingsDialog(QDialog):
    """
    Application settings dialog.

    Tabs:
    - Storage: Configure data storage locations
    - Display: Visual preferences
    - Session: Startup and session behavior
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)

        self.app_settings = get_settings()

        self._init_ui()
        self._load_settings()

    def _init_ui(self):
        """Initialize the dialog UI."""
        layout = QVBoxLayout(self)

        # Tab widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Create tabs
        self._create_storage_tab()
        self._create_display_tab()
        self._create_performance_tab()
        self._create_session_tab()

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel |
            QDialogButtonBox.StandardButton.Apply
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self._apply_settings)
        layout.addWidget(button_box)

    def _create_storage_tab(self):
        """Create the Storage settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Data Storage Group
        storage_group = QGroupBox("Data Storage Location")
        storage_layout = QFormLayout(storage_group)

        # Default storage directory
        storage_dir_layout = QHBoxLayout()
        self.storage_dir_edit = QLineEdit()
        self.storage_dir_edit.setPlaceholderText("Default: ~/.seisproc/data")
        self.storage_dir_edit.setToolTip(
            "Directory where imported SEG-Y files will be converted to Zarr/Parquet format.\n"
            "Leave empty to use default location."
        )
        storage_dir_layout.addWidget(self.storage_dir_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_storage_dir)
        storage_dir_layout.addWidget(browse_btn)

        storage_layout.addRow("Storage Directory:", storage_dir_layout)

        # Storage info label
        info_label = QLabel(
            "When you import a SEG-Y file, it is converted to an optimized\n"
            "Zarr/Parquet format for fast access. This directory stores those files."
        )
        info_label.setStyleSheet("color: gray; font-size: 10px;")
        storage_layout.addRow(info_label)

        layout.addWidget(storage_group)

        # Cache Settings Group
        cache_group = QGroupBox("Cache Settings")
        cache_layout = QFormLayout(cache_group)

        # Max cached datasets
        self.cache_limit_spin = QSpinBox()
        self.cache_limit_spin.setRange(1, 10)
        self.cache_limit_spin.setValue(3)
        self.cache_limit_spin.setToolTip(
            "Maximum number of datasets to keep in memory.\n"
            "Higher values use more RAM but allow faster switching."
        )
        cache_layout.addRow("Max Datasets in Memory:", self.cache_limit_spin)

        # Max cached gathers per dataset
        self.gather_cache_spin = QSpinBox()
        self.gather_cache_spin.setRange(1, 20)
        self.gather_cache_spin.setValue(5)
        self.gather_cache_spin.setToolTip(
            "Maximum number of gathers to keep cached per dataset.\n"
            "Higher values improve navigation speed."
        )
        cache_layout.addRow("Max Cached Gathers:", self.gather_cache_spin)

        layout.addWidget(cache_group)

        # Cleanup Group
        cleanup_group = QGroupBox("Storage Cleanup")
        cleanup_layout = QVBoxLayout(cleanup_group)

        cleanup_info = QLabel(
            "Imported data files can accumulate over time.\n"
            "Use the button below to view and clean up old data."
        )
        cleanup_info.setStyleSheet("color: gray; font-size: 10px;")
        cleanup_layout.addWidget(cleanup_info)

        cleanup_btn = QPushButton("Manage Storage...")
        cleanup_btn.clicked.connect(self._manage_storage)
        cleanup_layout.addWidget(cleanup_btn)

        layout.addWidget(cleanup_group)

        layout.addStretch()
        self.tabs.addTab(tab, "Storage")

    def _create_display_tab(self):
        """Create the Display settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Units Group
        units_group = QGroupBox("Coordinate Units")
        units_layout = QFormLayout(units_group)

        self.units_combo = QComboBox()
        self.units_combo.addItems(["Meters", "Feet"])
        self.units_combo.setToolTip("Units for coordinate display and trace spacing calculations")
        units_layout.addRow("Spatial Units:", self.units_combo)

        layout.addWidget(units_group)

        # Default Display Group
        display_group = QGroupBox("Default Display Settings")
        display_layout = QFormLayout(display_group)

        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["seismic", "gray", "viridis", "RdBu", "coolwarm"])
        self.colormap_combo.setToolTip("Default colormap for seismic display")
        display_layout.addRow("Default Colormap:", self.colormap_combo)

        self.interpolation_combo = QComboBox()
        self.interpolation_combo.addItems(["bilinear", "nearest", "bicubic"])
        self.interpolation_combo.setToolTip("Interpolation method for display")
        display_layout.addRow("Interpolation:", self.interpolation_combo)

        layout.addWidget(display_group)

        layout.addStretch()
        self.tabs.addTab(tab, "Display")

    def _create_performance_tab(self):
        """Create the Performance/GPU settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # GPU Status Group
        status_group = QGroupBox("GPU Status")
        status_layout = QVBoxLayout(status_group)

        self.gpu_status_label = QLabel()
        self.gpu_status_label.setStyleSheet("font-weight: bold;")
        status_layout.addWidget(self.gpu_status_label)

        self.gpu_info_label = QLabel()
        self.gpu_info_label.setStyleSheet("color: gray; font-size: 10px;")
        status_layout.addWidget(self.gpu_info_label)

        refresh_btn = QPushButton("Refresh GPU Status")
        refresh_btn.clicked.connect(self._refresh_gpu_status)
        status_layout.addWidget(refresh_btn)

        layout.addWidget(status_group)

        # GPU Settings Group
        gpu_group = QGroupBox("GPU Acceleration")
        gpu_layout = QFormLayout(gpu_group)

        self.gpu_enabled_checkbox = QCheckBox("Enable GPU acceleration")
        self.gpu_enabled_checkbox.setToolTip(
            "Use GPU for accelerated processing when available.\n"
            "Disable to force CPU-only processing."
        )
        self.gpu_enabled_checkbox.stateChanged.connect(self._on_gpu_enabled_changed)
        gpu_layout.addRow(self.gpu_enabled_checkbox)

        self.gpu_device_combo = QComboBox()
        self.gpu_device_combo.addItems(["Auto", "CUDA (NVIDIA)", "MPS (Apple)", "CPU Only"])
        self.gpu_device_combo.setToolTip(
            "Select preferred GPU device:\n"
            "• Auto: Automatically detect best available\n"
            "• CUDA: NVIDIA GPUs\n"
            "• MPS: Apple Silicon (M1/M2/M3)\n"
            "• CPU Only: Disable GPU"
        )
        gpu_layout.addRow("Device Preference:", self.gpu_device_combo)

        # GPU memory limit
        mem_layout = QHBoxLayout()
        self.gpu_memory_slider = QSlider(Qt.Orientation.Horizontal)
        self.gpu_memory_slider.setRange(10, 95)
        self.gpu_memory_slider.setValue(70)
        self.gpu_memory_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.gpu_memory_slider.setTickInterval(10)
        self.gpu_memory_slider.valueChanged.connect(self._on_memory_slider_changed)
        mem_layout.addWidget(self.gpu_memory_slider)

        self.gpu_memory_label = QLabel("70%")
        self.gpu_memory_label.setMinimumWidth(40)
        mem_layout.addWidget(self.gpu_memory_label)

        gpu_layout.addRow("GPU Memory Limit:", mem_layout)

        layout.addWidget(gpu_group)

        # Worker Settings Group
        workers_group = QGroupBox("Parallel Processing Workers")
        workers_layout = QVBoxLayout(workers_group)

        # Info label
        workers_info = QLabel(
            "Workers control parallel processing. With GPU enabled,\n"
            "fewer workers are optimal (GPU handles parallelism internally)."
        )
        workers_info.setStyleSheet("color: gray; font-size: 10px;")
        workers_layout.addWidget(workers_info)

        # GPU workers
        gpu_workers_layout = QHBoxLayout()
        self.gpu_workers_auto_checkbox = QCheckBox("Auto")
        self.gpu_workers_auto_checkbox.setToolTip("Automatically calculate optimal worker count for GPU mode")
        self.gpu_workers_auto_checkbox.stateChanged.connect(self._on_gpu_workers_auto_changed)
        gpu_workers_layout.addWidget(self.gpu_workers_auto_checkbox)

        self.gpu_workers_spin = QSpinBox()
        self.gpu_workers_spin.setRange(1, 16)
        self.gpu_workers_spin.setValue(1)
        self.gpu_workers_spin.setToolTip(
            "Number of parallel workers for GPU processing.\n"
            "Recommended: 1 (GPU handles parallelism internally)"
        )
        gpu_workers_layout.addWidget(self.gpu_workers_spin)

        self.gpu_workers_recommended_label = QLabel()
        self.gpu_workers_recommended_label.setStyleSheet("color: gray; font-size: 10px;")
        gpu_workers_layout.addWidget(self.gpu_workers_recommended_label)
        gpu_workers_layout.addStretch()

        workers_form = QFormLayout()
        workers_form.addRow("GPU Mode Workers:", gpu_workers_layout)

        # CPU workers
        cpu_workers_layout = QHBoxLayout()
        self.cpu_workers_auto_checkbox = QCheckBox("Auto")
        self.cpu_workers_auto_checkbox.setToolTip("Automatically calculate optimal worker count for CPU mode")
        self.cpu_workers_auto_checkbox.stateChanged.connect(self._on_cpu_workers_auto_changed)
        cpu_workers_layout.addWidget(self.cpu_workers_auto_checkbox)

        self.cpu_workers_spin = QSpinBox()
        self.cpu_workers_spin.setRange(1, 64)
        self.cpu_workers_spin.setValue(4)
        self.cpu_workers_spin.setToolTip(
            "Number of parallel workers for CPU-only processing.\n"
            "Recommended: CPU cores - 1"
        )
        cpu_workers_layout.addWidget(self.cpu_workers_spin)

        self.cpu_workers_recommended_label = QLabel()
        self.cpu_workers_recommended_label.setStyleSheet("color: gray; font-size: 10px;")
        cpu_workers_layout.addWidget(self.cpu_workers_recommended_label)
        cpu_workers_layout.addStretch()

        workers_form.addRow("CPU Mode Workers:", cpu_workers_layout)
        workers_layout.addLayout(workers_form)

        layout.addWidget(workers_group)

        # Update recommended labels
        self._update_recommended_labels()

        layout.addStretch()
        self.tabs.addTab(tab, "Performance")

    def _refresh_gpu_status(self):
        """Refresh GPU status display."""
        if not TORCH_AVAILABLE:
            self.gpu_status_label.setText("⚠️ PyTorch not installed")
            self.gpu_info_label.setText("Install PyTorch for GPU acceleration")
            return

        try:
            cuda_available = torch.cuda.is_available()
            mps_available = torch.backends.mps.is_available()

            if cuda_available:
                device_name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                mem_total = props.total_memory / (1024**3)
                mem_used = torch.cuda.memory_allocated(0) / (1024**3)
                self.gpu_status_label.setText(f"🟢 GPU Available: {device_name}")
                self.gpu_info_label.setText(
                    f"Memory: {mem_used:.1f} / {mem_total:.1f} GB | "
                    f"CUDA {torch.version.cuda} | PyTorch {torch.__version__}"
                )
            elif mps_available:
                self.gpu_status_label.setText("🟢 GPU Available: Apple Silicon (MPS)")
                self.gpu_info_label.setText(f"PyTorch {torch.__version__} with Metal backend")
            else:
                self.gpu_status_label.setText("🟡 No GPU Available")
                self.gpu_info_label.setText("Processing will use CPU only")
        except Exception as e:
            self.gpu_status_label.setText("⚠️ GPU Detection Error")
            self.gpu_info_label.setText(str(e))

    def _on_gpu_enabled_changed(self, state):
        """Handle GPU enabled checkbox change."""
        enabled = state == Qt.CheckState.Checked.value
        self.gpu_device_combo.setEnabled(enabled)
        self.gpu_memory_slider.setEnabled(enabled)
        self.gpu_workers_auto_checkbox.setEnabled(enabled)
        self.gpu_workers_spin.setEnabled(enabled and not self.gpu_workers_auto_checkbox.isChecked())

    def _on_memory_slider_changed(self, value):
        """Update memory limit label."""
        self.gpu_memory_label.setText(f"{value}%")

    def _on_gpu_workers_auto_changed(self, state):
        """Handle GPU workers auto checkbox change."""
        auto = state == Qt.CheckState.Checked.value
        self.gpu_workers_spin.setEnabled(not auto)
        if auto:
            self.gpu_workers_spin.setValue(self.app_settings.get_recommended_gpu_workers())

    def _on_cpu_workers_auto_changed(self, state):
        """Handle CPU workers auto checkbox change."""
        auto = state == Qt.CheckState.Checked.value
        self.cpu_workers_spin.setEnabled(not auto)
        if auto:
            self.cpu_workers_spin.setValue(self.app_settings.get_recommended_cpu_workers())

    def _update_recommended_labels(self):
        """Update recommended worker count labels."""
        gpu_rec = self.app_settings.get_recommended_gpu_workers()
        cpu_rec = self.app_settings.get_recommended_cpu_workers()

        self.gpu_workers_recommended_label.setText(f"(Recommended: {gpu_rec})")
        self.cpu_workers_recommended_label.setText(f"(Recommended: {cpu_rec})")

    def _create_session_tab(self):
        """Create the Session settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Startup Group
        startup_group = QGroupBox("Startup Behavior")
        startup_layout = QVBoxLayout(startup_group)

        self.auto_load_checkbox = QCheckBox("Restore last session on startup")
        self.auto_load_checkbox.setToolTip(
            "When enabled, the application will try to restore the last\n"
            "loaded dataset and viewport settings when starting."
        )
        startup_layout.addWidget(self.auto_load_checkbox)

        self.remember_window_checkbox = QCheckBox("Remember window size and position")
        self.remember_window_checkbox.setToolTip("Restore window geometry on startup")
        startup_layout.addWidget(self.remember_window_checkbox)

        layout.addWidget(startup_group)

        # Recent Files Group
        recent_group = QGroupBox("Recent Files")
        recent_layout = QFormLayout(recent_group)

        self.max_recent_spin = QSpinBox()
        self.max_recent_spin.setRange(5, 50)
        self.max_recent_spin.setValue(10)
        self.max_recent_spin.setToolTip("Maximum number of recent files to remember")
        recent_layout.addRow("Max Recent Files:", self.max_recent_spin)

        clear_recent_btn = QPushButton("Clear Recent Files")
        clear_recent_btn.clicked.connect(self._clear_recent_files)
        recent_layout.addRow(clear_recent_btn)

        layout.addWidget(recent_group)

        # Session Data Group
        session_group = QGroupBox("Session Data")
        session_layout = QVBoxLayout(session_group)

        clear_session_btn = QPushButton("Clear Saved Session")
        clear_session_btn.setToolTip("Clear saved viewport and navigation state")
        clear_session_btn.clicked.connect(self._clear_session)
        session_layout.addWidget(clear_session_btn)

        reset_all_btn = QPushButton("Reset All Settings to Defaults")
        reset_all_btn.setToolTip("Reset all settings to factory defaults")
        reset_all_btn.clicked.connect(self._reset_all_settings)
        session_layout.addWidget(reset_all_btn)

        layout.addWidget(session_group)

        layout.addStretch()
        self.tabs.addTab(tab, "Session")

    def _load_settings(self):
        """Load current settings into the UI."""
        # Storage settings
        storage_dir = self.app_settings.get_storage_directory()
        if storage_dir:
            self.storage_dir_edit.setText(str(storage_dir))

        self.cache_limit_spin.setValue(self.app_settings.get_dataset_cache_limit())
        self.gather_cache_spin.setValue(self.app_settings.get_gather_cache_limit())

        # Display settings
        units = self.app_settings.get_spatial_units()
        self.units_combo.setCurrentText("Meters" if units == "meters" else "Feet")

        session = self.app_settings.get_session_state()
        colormap = session.get('colormap', 'seismic')
        idx = self.colormap_combo.findText(colormap)
        if idx >= 0:
            self.colormap_combo.setCurrentIndex(idx)

        interpolation = session.get('interpolation', 'bilinear')
        idx = self.interpolation_combo.findText(interpolation)
        if idx >= 0:
            self.interpolation_combo.setCurrentIndex(idx)

        # Performance/GPU settings
        self._refresh_gpu_status()

        self.gpu_enabled_checkbox.setChecked(self.app_settings.get_gpu_enabled())

        device_pref = self.app_settings.get_gpu_device_preference()
        device_map = {'auto': 0, 'cuda': 1, 'mps': 2, 'cpu': 3}
        self.gpu_device_combo.setCurrentIndex(device_map.get(device_pref, 0))

        self.gpu_memory_slider.setValue(self.app_settings.get_gpu_memory_limit_percent())
        self.gpu_memory_label.setText(f"{self.gpu_memory_slider.value()}%")

        self.gpu_workers_auto_checkbox.setChecked(self.app_settings.get_processing_workers_auto())
        self.gpu_workers_spin.setValue(self.app_settings.get_processing_workers())
        self.gpu_workers_spin.setEnabled(not self.gpu_workers_auto_checkbox.isChecked())

        self.cpu_workers_auto_checkbox.setChecked(self.app_settings.get_cpu_workers_auto())
        self.cpu_workers_spin.setValue(self.app_settings.get_cpu_workers())
        self.cpu_workers_spin.setEnabled(not self.cpu_workers_auto_checkbox.isChecked())

        # Update enabled state based on GPU checkbox
        gpu_enabled = self.gpu_enabled_checkbox.isChecked()
        self.gpu_device_combo.setEnabled(gpu_enabled)
        self.gpu_memory_slider.setEnabled(gpu_enabled)
        self.gpu_workers_auto_checkbox.setEnabled(gpu_enabled)

        # Session settings
        self.auto_load_checkbox.setChecked(self.app_settings.get_auto_load_last_dataset())
        self.remember_window_checkbox.setChecked(self.app_settings.get_remember_window_geometry())
        self.max_recent_spin.setValue(self.app_settings.get_max_recent_files())

    def _apply_settings(self):
        """Apply settings without closing the dialog."""
        # Storage settings
        storage_dir = self.storage_dir_edit.text().strip()
        if storage_dir:
            path = Path(storage_dir)
            if not path.exists():
                reply = QMessageBox.question(
                    self,
                    "Create Directory",
                    f"Directory does not exist:\n{storage_dir}\n\nCreate it?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    try:
                        path.mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        QMessageBox.critical(self, "Error", f"Failed to create directory:\n{e}")
                        return
                else:
                    return
            self.app_settings.set_storage_directory(storage_dir)
        else:
            self.app_settings.set_storage_directory(None)

        self.app_settings.set_dataset_cache_limit(self.cache_limit_spin.value())
        self.app_settings.set_gather_cache_limit(self.gather_cache_spin.value())

        # Display settings
        units = "meters" if self.units_combo.currentText() == "Meters" else "feet"
        self.app_settings.set_spatial_units(units)

        self.app_settings.save_display_state(
            colormap=self.colormap_combo.currentText(),
            interpolation=self.interpolation_combo.currentText()
        )

        # Performance/GPU settings
        self.app_settings.set_gpu_enabled(self.gpu_enabled_checkbox.isChecked())

        device_map = {0: 'auto', 1: 'cuda', 2: 'mps', 3: 'cpu'}
        device_pref = device_map.get(self.gpu_device_combo.currentIndex(), 'auto')
        self.app_settings.set_gpu_device_preference(device_pref)

        self.app_settings.set_gpu_memory_limit_percent(self.gpu_memory_slider.value())

        self.app_settings.set_processing_workers_auto(self.gpu_workers_auto_checkbox.isChecked())
        self.app_settings.set_processing_workers(self.gpu_workers_spin.value())

        self.app_settings.set_cpu_workers_auto(self.cpu_workers_auto_checkbox.isChecked())
        self.app_settings.set_cpu_workers(self.cpu_workers_spin.value())

        # Session settings
        self.app_settings.set_auto_load_last_dataset(self.auto_load_checkbox.isChecked())
        self.app_settings.set_remember_window_geometry(self.remember_window_checkbox.isChecked())
        self.app_settings.set_max_recent_files(self.max_recent_spin.value())

        logger.info("Settings applied")

    def _on_accept(self):
        """Handle OK button - apply and close."""
        self._apply_settings()
        self.accept()

    def _browse_storage_dir(self):
        """Open directory browser for storage location."""
        current = self.storage_dir_edit.text()
        if not current:
            current = str(Path.home())

        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Storage Directory",
            current
        )

        if directory:
            self.storage_dir_edit.setText(directory)

    def _manage_storage(self):
        """Open storage management dialog."""
        storage_dir = self.app_settings.get_storage_directory()
        if not storage_dir:
            storage_dir = self.app_settings.get_default_storage_directory()

        if not Path(storage_dir).exists():
            QMessageBox.information(
                self,
                "No Data",
                f"Storage directory does not exist yet:\n{storage_dir}\n\n"
                "It will be created when you import your first SEG-Y file."
            )
            return

        # Calculate storage usage
        total_size = 0
        file_count = 0
        try:
            for item in Path(storage_dir).rglob('*'):
                if item.is_file():
                    total_size += item.stat().st_size
                    file_count += 1
        except Exception as e:
            logger.error(f"Error scanning storage: {e}")

        size_mb = total_size / (1024 * 1024)
        size_gb = total_size / (1024 * 1024 * 1024)

        if size_gb >= 1:
            size_str = f"{size_gb:.2f} GB"
        else:
            size_str = f"{size_mb:.1f} MB"

        msg = (f"Storage Location:\n{storage_dir}\n\n"
               f"Total Size: {size_str}\n"
               f"Files: {file_count:,}\n\n"
               "To free space, you can manually delete dataset folders\n"
               "that you no longer need.")

        QMessageBox.information(self, "Storage Usage", msg)

    def _clear_recent_files(self):
        """Clear the recent files list."""
        reply = QMessageBox.question(
            self,
            "Clear Recent Files",
            "Clear the list of recently opened files?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.app_settings.clear_recent_files()
            QMessageBox.information(self, "Done", "Recent files list cleared.")

    def _clear_session(self):
        """Clear saved session state."""
        reply = QMessageBox.question(
            self,
            "Clear Session",
            "Clear saved session state (viewport, navigation)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.app_settings.clear_session_state()
            QMessageBox.information(self, "Done", "Session state cleared.")

    def _reset_all_settings(self):
        """Reset all settings to defaults."""
        reply = QMessageBox.warning(
            self,
            "Reset All Settings",
            "This will reset ALL settings to their default values.\n\n"
            "This action cannot be undone. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.app_settings.reset_to_defaults()
            self._load_settings()  # Reload UI
            QMessageBox.information(self, "Done", "All settings reset to defaults.")
