"""
Volume Header Selection Dialog

Dialog for selecting headers to build a 3D volume from 2D gathers.
User selects inline and crossline header keys.
"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QFormLayout, QMessageBox,
    QProgressDialog, QApplication
)
from PyQt6.QtCore import Qt
import pandas as pd
from typing import Optional, Tuple, List
import numpy as np

from utils.volume_builder import (
    get_available_volume_headers,
    estimate_volume_size,
    build_volume_from_gathers,
    VolumeGeometry
)
from models.seismic_volume import SeismicVolume


class VolumeHeaderDialog(QDialog):
    """
    Dialog for selecting headers to build a 3D volume.

    User selects which header to use for inline (X) and crossline (Y) axes.
    Shows preview of resulting volume size and coverage.
    """

    def __init__(
        self,
        headers_df: pd.DataFrame,
        n_samples: int,
        parent=None
    ):
        """
        Initialize dialog.

        Args:
            headers_df: DataFrame with trace headers
            n_samples: Number of time samples per trace
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Build 3D Volume - Select Headers")
        self.setMinimumWidth(450)

        self.headers_df = headers_df
        self.n_samples = n_samples
        self.selected_inline = None
        self.selected_xline = None

        self._init_ui()
        self._update_preview()

    def _get_all_headers_with_counts(self) -> List[Tuple[str, int]]:
        """Get all headers from DataFrame with their unique value counts."""
        import logging
        logger = logging.getLogger(__name__)

        if self.headers_df is None or self.headers_df.empty:
            logger.warning("headers_df is None or empty")
            return []

        logger.info(f"DataFrame has {len(self.headers_df)} rows, columns: {list(self.headers_df.columns)}")

        headers_with_counts = []
        for col in self.headers_df.columns:
            try:
                n_unique = self.headers_df[col].nunique()
                logger.debug(f"Column '{col}': {n_unique} unique values")
                headers_with_counts.append((col, n_unique))
            except Exception as e:
                logger.warning(f"Error checking column '{col}': {e}")

        # Sort by unique count descending (most useful first), but keep usable ones at top
        headers_with_counts.sort(key=lambda x: (-1 if x[1] >= 2 else 0, -x[1]))

        logger.info(f"All headers: {[(h, c) for h, c in headers_with_counts]}")
        return headers_with_counts

    def _init_ui(self):
        """Initialize user interface."""
        layout = QVBoxLayout(self)

        # Info label
        info = QLabel(
            "Select header keys to define the 3D volume axes.\n"
            "Inline = X direction, Crossline = Y direction."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #666; margin-bottom: 10px;")
        layout.addWidget(info)

        # Get ALL headers from the DataFrame with unique counts
        self.headers_with_counts = self._get_all_headers_with_counts()
        # Filter to usable headers (2+ unique values)
        self.available_headers = [h for h, c in self.headers_with_counts if c >= 2]
        # Create display items showing count: "header_name (N values)"
        self.header_display_items = [f"{h} ({c} values)" for h, c in self.headers_with_counts if c >= 2]
        # Map display to actual header name
        self.display_to_header = {f"{h} ({c} values)": h for h, c in self.headers_with_counts if c >= 2}

        # Create ALL widgets FIRST before connecting any signals
        self.preview_label = QLabel("Calculating...")
        self.preview_label.setStyleSheet("font-family: monospace;")

        self.warning_label = QLabel("")
        self.warning_label.setStyleSheet("color: orange;")
        self.warning_label.setWordWrap(True)
        self.warning_label.hide()

        self.build_btn = QPushButton("Build Volume")
        self.build_btn.setStyleSheet("font-weight: bold;")
        self.build_btn.setEnabled(len(self.available_headers) >= 2)

        # Header selection group
        header_group = QGroupBox("Header Selection")
        form_layout = QFormLayout(header_group)

        # Show info about filtered headers
        n_filtered = len([h for h, c in self.headers_with_counts if c < 2])
        if n_filtered > 0:
            filtered_names = [h for h, c in self.headers_with_counts if c < 2]
            info_label = QLabel(
                f"Note: {n_filtered} headers hidden (only 1 unique value):\n"
                f"{', '.join(filtered_names[:5])}{'...' if len(filtered_names) > 5 else ''}"
            )
            info_label.setStyleSheet("color: #888; font-size: 10px;")
            info_label.setWordWrap(True)
            form_layout.addRow(info_label)

        if not self.available_headers:
            # No suitable headers found
            error_label = QLabel(
                "No suitable headers found for volume building.\n"
                "Need at least 2 headers with multiple unique values."
            )
            error_label.setStyleSheet("color: red;")
            form_layout.addRow(error_label)
        else:
            # Inline (X) selection - block signals during setup
            self.inline_combo = QComboBox()
            self.inline_combo.blockSignals(True)
            self.inline_combo.addItems(self.header_display_items)

            # Try to set a sensible default
            defaults_inline = ['sin', 'field_record', 's_line', 'SourceLine', 'Inline', 'CDP', 'FFID']
            for default in defaults_inline:
                for display_item in self.header_display_items:
                    if display_item.startswith(default + " ("):
                        self.inline_combo.setCurrentText(display_item)
                        break
                else:
                    continue
                break

            self.inline_combo.blockSignals(False)
            self.inline_combo.currentIndexChanged.connect(self._update_preview)
            form_layout.addRow("Inline (X) Key:", self.inline_combo)

            # Crossline (Y) selection - block signals during setup
            self.xline_combo = QComboBox()
            self.xline_combo.blockSignals(True)
            self.xline_combo.addItems(self.header_display_items)

            # Try to set a sensible default different from inline
            defaults_xline = ['rec_sloc', 'trace_number', 'ReceiverStation', 'Crossline', 'Channel']
            for default in defaults_xline:
                for display_item in self.header_display_items:
                    if display_item.startswith(default + " ("):
                        if display_item != self.inline_combo.currentText():
                            self.xline_combo.setCurrentText(display_item)
                            break
                else:
                    continue
                break

            # If xline is same as inline, pick next available
            if self.xline_combo.currentText() == self.inline_combo.currentText():
                for display_item in self.header_display_items:
                    if display_item != self.inline_combo.currentText():
                        self.xline_combo.setCurrentText(display_item)
                        break

            self.xline_combo.blockSignals(False)
            self.xline_combo.currentIndexChanged.connect(self._update_preview)
            form_layout.addRow("Crossline (Y) Key:", self.xline_combo)

        layout.addWidget(header_group)

        # Preview group
        preview_group = QGroupBox("Volume Preview")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.addWidget(self.preview_label)
        layout.addWidget(preview_group)

        # Warning label
        layout.addWidget(self.warning_label)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        self.build_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.build_btn)

        layout.addLayout(button_layout)

    def _update_preview(self):
        """Update the volume size preview."""
        # Safety guards - widgets may not exist yet during init
        if not hasattr(self, 'preview_label') or not hasattr(self, 'inline_combo') or not hasattr(self, 'xline_combo'):
            return

        # Get actual header names from display text
        inline_display = self.inline_combo.currentText()
        xline_display = self.xline_combo.currentText()
        inline_key = self.display_to_header.get(inline_display, inline_display)
        xline_key = self.display_to_header.get(xline_display, xline_display)

        if inline_display == xline_display:
            self.preview_label.setText("Error: Inline and Crossline must be different")
            self.build_btn.setEnabled(False)
            return

        try:
            estimate = estimate_volume_size(
                self.headers_df,
                inline_key,
                xline_key,
                self.n_samples
            )

            preview_text = (
                f"Shape: {estimate['n_samples']} x {estimate['n_inlines']} x {estimate['n_xlines']}\n"
                f"       (samples x inlines x xlines)\n\n"
                f"Size: {estimate['size_mb']:.1f} MB ({estimate['size_gb']:.2f} GB)\n\n"
                f"Traces: {estimate['n_traces']:,} available\n"
                f"        {estimate['theoretical_traces']:,} theoretical\n"
                f"        {estimate['coverage_percent']:.1f}% coverage\n\n"
                f"Inline range: {estimate['inline_range'][0]} - {estimate['inline_range'][1]}\n"
                f"Xline range: {estimate['xline_range'][0]} - {estimate['xline_range'][1]}"
            )
            self.preview_label.setText(preview_text)

            # Check for warnings
            warnings = []
            if estimate['size_mb'] > 2000:
                warnings.append(f"Large volume ({estimate['size_gb']:.1f} GB) may require significant memory")
            if estimate['coverage_percent'] < 50:
                warnings.append(f"Low coverage ({estimate['coverage_percent']:.0f}%) - many traces may be missing")
            if estimate['n_inlines'] < 4 or estimate['n_xlines'] < 4:
                warnings.append("Very small volume - may not be suitable for FKK filtering")

            if warnings:
                self.warning_label.setText("\n".join(warnings))
                self.warning_label.show()
            else:
                self.warning_label.hide()

            self.build_btn.setEnabled(True)

        except Exception as e:
            self.preview_label.setText(f"Error: {str(e)}")
            self.build_btn.setEnabled(False)

    def get_selected_headers(self) -> Tuple[str, str]:
        """
        Get selected inline and crossline header keys.

        Returns:
            Tuple of (inline_key, xline_key)
        """
        inline_display = self.inline_combo.currentText()
        xline_display = self.xline_combo.currentText()
        return (
            self.display_to_header.get(inline_display, inline_display),
            self.display_to_header.get(xline_display, xline_display)
        )


def build_volume_with_dialog(
    traces_data: np.ndarray,
    headers_df: pd.DataFrame,
    sample_rate_ms: float,
    coordinate_units: str = 'meters',
    parent=None
) -> Optional[Tuple[SeismicVolume, VolumeGeometry]]:
    """
    Show dialog to select headers and build 3D volume.

    Args:
        traces_data: 2D array of traces (n_samples, n_traces)
        headers_df: DataFrame with trace headers
        sample_rate_ms: Sample rate in milliseconds
        coordinate_units: 'meters' or 'feet'
        parent: Parent widget

    Returns:
        Tuple of (SeismicVolume, VolumeGeometry) or None if cancelled
    """
    n_samples = traces_data.shape[0]

    # Show header selection dialog
    dialog = VolumeHeaderDialog(headers_df, n_samples, parent)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None

    inline_key, xline_key = dialog.get_selected_headers()

    # Show progress dialog
    progress = QProgressDialog(
        "Building 3D volume...",
        "Cancel",
        0, 100,
        parent
    )
    progress.setWindowModality(Qt.WindowModality.WindowModal)
    progress.setMinimumDuration(500)

    def update_progress(pct, msg):
        progress.setValue(int(pct))
        progress.setLabelText(msg)
        QApplication.processEvents()
        return not progress.wasCanceled()

    try:
        volume, geometry = build_volume_from_gathers(
            traces_data=traces_data,
            headers_df=headers_df,
            inline_key=inline_key,
            xline_key=xline_key,
            sample_rate_ms=sample_rate_ms,
            coordinate_units=coordinate_units,
            progress_callback=update_progress
        )

        progress.setValue(100)
        return volume, geometry

    except Exception as e:
        progress.close()
        QMessageBox.critical(
            parent,
            "Error",
            f"Failed to build 3D volume:\n\n{str(e)}"
        )
        return None
