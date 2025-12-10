"""
Volume Header Selection Dialog

Dialog for selecting headers to build a 3D volume from 2D gathers.
User selects inline and crossline header keys.

Features:
- Automatic distance calculation from coordinate headers
- User can override calculated dx/dy values
- Shows distance source (coordinates, header_diff, or default)
"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QFormLayout, QMessageBox,
    QProgressDialog, QApplication, QDoubleSpinBox
)
from PyQt6.QtCore import Qt
import pandas as pd
from typing import Optional, Tuple, List
import numpy as np

from utils.volume_builder import (
    get_available_volume_headers,
    estimate_volume_size,
    estimate_volume_size_fast,
    build_volume_from_gathers,
    calculate_spatial_distances,
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
        coord_scalar: float = 1.0,
        parent=None
    ):
        """
        Initialize dialog.

        Args:
            headers_df: DataFrame with trace headers
            n_samples: Number of time samples per trace
            coord_scalar: Coordinate scalar from SEG-Y (e.g., -100 means divide by 100)
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Build 3D Volume - Select Headers")
        self.setMinimumWidth(500)

        self.headers_df = headers_df
        self.n_samples = n_samples
        self.coord_scalar = coord_scalar
        self.selected_inline = None
        self.selected_xline = None

        # Store calculated distances
        self._calculated_dx = 25.0
        self._calculated_dy = 25.0
        self._dx_source = 'default'
        self._dy_source = 'default'

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

            # Auto-detect button
            self.auto_btn = QPushButton("Auto-detect Best")
            self.auto_btn.setToolTip("Find header combination with best coverage")
            self.auto_btn.clicked.connect(self._on_auto_detect)
            form_layout.addRow("", self.auto_btn)

        layout.addWidget(header_group)

        # Spatial Distance Group - for FKK filter accuracy
        distance_group = QGroupBox("Spatial Distances (for FKK Filter)")
        distance_layout = QFormLayout(distance_group)

        # Info about distance calculation
        dist_info = QLabel(
            "Distances are calculated from coordinates.\n"
            "Adjust if incorrect for proper FKK velocity filtering."
        )
        dist_info.setStyleSheet("color: #666; font-size: 10px;")
        dist_info.setWordWrap(True)
        distance_layout.addRow(dist_info)

        # dx spinbox (inline spacing)
        dx_layout = QHBoxLayout()
        self.dx_spin = QDoubleSpinBox()
        self.dx_spin.setRange(1.0, 1000.0)
        self.dx_spin.setValue(25.0)
        self.dx_spin.setSuffix(" m")
        self.dx_spin.setDecimals(1)
        self.dx_spin.setToolTip("Inline spacing in meters (X direction)")
        dx_layout.addWidget(self.dx_spin)
        self.dx_source_label = QLabel("(default)")
        self.dx_source_label.setStyleSheet("color: #888; font-size: 10px;")
        dx_layout.addWidget(self.dx_source_label)
        distance_layout.addRow("dx (inline):", dx_layout)

        # dy spinbox (crossline spacing)
        dy_layout = QHBoxLayout()
        self.dy_spin = QDoubleSpinBox()
        self.dy_spin.setRange(1.0, 1000.0)
        self.dy_spin.setValue(25.0)
        self.dy_spin.setSuffix(" m")
        self.dy_spin.setDecimals(1)
        self.dy_spin.setToolTip("Crossline spacing in meters (Y direction)")
        dy_layout.addWidget(self.dy_spin)
        self.dy_source_label = QLabel("(default)")
        self.dy_source_label.setStyleSheet("color: #888; font-size: 10px;")
        dy_layout.addWidget(self.dy_source_label)
        distance_layout.addRow("dy (crossline):", dy_layout)

        # Recalculate button
        self.recalc_dist_btn = QPushButton("Recalculate from Coordinates")
        self.recalc_dist_btn.setToolTip("Recalculate distances from coordinate headers")
        self.recalc_dist_btn.clicked.connect(self._recalculate_distances)
        distance_layout.addRow("", self.recalc_dist_btn)

        layout.addWidget(distance_group)

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

    def _recalculate_distances(self):
        """Recalculate distances from coordinates and update spinboxes."""
        if not hasattr(self, 'inline_combo') or not hasattr(self, 'xline_combo'):
            return

        inline_display = self.inline_combo.currentText()
        xline_display = self.xline_combo.currentText()
        inline_key = self.display_to_header.get(inline_display, inline_display)
        xline_key = self.display_to_header.get(xline_display, xline_display)

        distances = calculate_spatial_distances(
            self.headers_df, inline_key, xline_key, self.coord_scalar
        )

        self._calculated_dx = distances['dx']
        self._calculated_dy = distances['dy']
        self._dx_source = distances['dx_source']
        self._dy_source = distances['dy_source']

        # Update spinboxes only if values were calculated
        self.dx_spin.blockSignals(True)
        self.dy_spin.blockSignals(True)
        if distances['dx'] is not None:
            self.dx_spin.setValue(distances['dx'])
        if distances['dy'] is not None:
            self.dy_spin.setValue(distances['dy'])
        self.dx_spin.blockSignals(False)
        self.dy_spin.blockSignals(False)

        # Update source labels
        source_style_ok = "color: green; font-size: 10px;"
        source_style_error = "color: red; font-size: 10px; font-weight: bold;"

        if distances['dx_source'] is not None:
            self.dx_source_label.setText(f"(from {distances['dx_source']})")
            self.dx_source_label.setStyleSheet(source_style_ok)
        else:
            self.dx_source_label.setText("(ENTER VALUE)")
            self.dx_source_label.setStyleSheet(source_style_error)

        if distances['dy_source'] is not None:
            self.dy_source_label.setText(f"(from {distances['dy_source']})")
            self.dy_source_label.setStyleSheet(source_style_ok)
        else:
            self.dy_source_label.setText("(ENTER VALUE)")
            self.dy_source_label.setStyleSheet(source_style_error)

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
                self.n_samples,
                coord_scalar=self.coord_scalar
            )

            # Update distance spinboxes from estimate
            self._calculated_dx = estimate['dx']
            self._calculated_dy = estimate['dy']
            self._dx_source = estimate['dx_source']
            self._dy_source = estimate['dy_source']

            # Update spinboxes if values were calculated
            if hasattr(self, 'dx_spin'):
                self.dx_spin.blockSignals(True)
                self.dy_spin.blockSignals(True)

                # Only update spinbox if we got a calculated value
                if estimate['dx'] is not None:
                    self.dx_spin.setValue(estimate['dx'])
                if estimate['dy'] is not None:
                    self.dy_spin.setValue(estimate['dy'])

                self.dx_spin.blockSignals(False)
                self.dy_spin.blockSignals(False)

                # Update source labels
                source_style_ok = "color: green; font-size: 10px;"
                source_style_error = "color: red; font-size: 10px; font-weight: bold;"

                if estimate['dx_source'] is not None:
                    self.dx_source_label.setText(f"(from {estimate['dx_source']})")
                    self.dx_source_label.setStyleSheet(source_style_ok)
                else:
                    self.dx_source_label.setText("(ENTER VALUE)")
                    self.dx_source_label.setStyleSheet(source_style_error)

                if estimate['dy_source'] is not None:
                    self.dy_source_label.setText(f"(from {estimate['dy_source']})")
                    self.dy_source_label.setStyleSheet(source_style_ok)
                else:
                    self.dy_source_label.setText("(ENTER VALUE)")
                    self.dy_source_label.setStyleSheet(source_style_error)

            # Build preview text
            preview_text = (
                f"Shape: {estimate['n_samples']} x {estimate['n_inlines']} x {estimate['n_xlines']}\n"
                f"       (samples × inlines × xlines)\n\n"
                f"Size: {estimate['size_mb']:.1f} MB ({estimate['size_gb']:.2f} GB)\n\n"
                f"Traces: {estimate['n_traces']:,} input\n"
                f"        {estimate['n_unique_positions']:,} unique positions\n"
                f"        {estimate['theoretical_traces']:,} grid cells ({estimate['n_inlines']}×{estimate['n_xlines']})\n"
                f"        {estimate['coverage_percent']:.1f}% grid coverage\n"
            )

            # Show duplicate info if any
            if estimate['n_duplicates'] > 0:
                preview_text += f"        {estimate['n_duplicates']:,} duplicate positions (will be overwritten)\n"

            preview_text += (
                f"\nInline range: {estimate['inline_range'][0]} - {estimate['inline_range'][1]}\n"
                f"Xline range: {estimate['xline_range'][0]} - {estimate['xline_range'][1]}"
            )

            # Add spacing info
            dx_str = f"{estimate['dx']:.1f}m" if estimate['dx'] is not None else "NOT SET"
            dy_str = f"{estimate['dy']:.1f}m" if estimate['dy'] is not None else "NOT SET"
            preview_text += f"\n\nSpacing: dx={dx_str}, dy={dy_str}"

            self.preview_label.setText(preview_text)

            # Check for warnings
            warnings = []
            if estimate['size_mb'] > 2000:
                warnings.append(f"⚠ Large volume ({estimate['size_gb']:.1f} GB) may require significant memory")

            # Coverage warning - check both grid coverage and trace usage
            grid_coverage = estimate['coverage_percent']
            trace_usage = (estimate['n_unique_positions'] / estimate['n_traces']) * 100 if estimate['n_traces'] > 0 else 0

            if grid_coverage < 50:
                warnings.append(f"⚠ Low grid coverage ({grid_coverage:.0f}%) - many grid cells are empty")
            if trace_usage < 90:
                warnings.append(f"⚠ Only {trace_usage:.0f}% of traces mapped - check header selection")
            if estimate['n_duplicates'] > estimate['n_traces'] * 0.1:
                warnings.append(f"⚠ Many duplicate positions ({estimate['n_duplicates']}) - traces may be overwritten")

            if estimate['n_inlines'] < 4 or estimate['n_xlines'] < 4:
                warnings.append("⚠ Very small volume - may not be suitable for FKK filtering")

            # Suggest better combination if coverage is low
            if grid_coverage < 80 and trace_usage < 90:
                best = self._find_best_header_combination()
                if best and (best['inline'] != inline_key or best['xline'] != xline_key):
                    warnings.append(
                        f"💡 Suggested: {best['inline']} × {best['xline']} "
                        f"({best['coverage']:.0f}% coverage)"
                    )

            if warnings:
                self.warning_label.setText("\n".join(warnings))
                self.warning_label.show()
            else:
                self.warning_label.hide()

            self.build_btn.setEnabled(True)

        except Exception as e:
            self.preview_label.setText(f"Error: {str(e)}")
            self.build_btn.setEnabled(False)

    def _find_best_header_combination(self):
        """
        Find header combination that gives best coverage.

        Scoring prioritizes:
        1. High coverage percentage (traces that map to unique grid positions)
        2. Low number of duplicates (traces overwriting each other)
        3. Grid size close to trace count (efficient use of memory)
        """
        import logging
        logger = logging.getLogger(__name__)

        if len(self.available_headers) < 2:
            return None

        n_traces = len(self.headers_df)
        best_score = -1
        best_combo = None

        logger.info(f"Auto-detecting best headers for {n_traces} traces from {len(self.available_headers)} available headers")

        # Check all combinations (limit to prevent excessive computation)
        headers_to_check = self.available_headers[:15]

        for i, h1 in enumerate(headers_to_check):
            for h2 in headers_to_check[i+1:]:
                try:
                    # Use fast estimate (no distance calculation)
                    estimate = estimate_volume_size_fast(
                        self.headers_df, h1, h2, self.n_samples
                    )

                    # Calculate score based on multiple factors:
                    # 1. Coverage: what % of grid is filled
                    coverage = estimate['coverage_percent']

                    # 2. Efficiency: what % of traces map to unique positions (no duplicates)
                    unique_ratio = (estimate['n_unique_positions'] / n_traces) * 100 if n_traces > 0 else 0

                    # 3. Grid efficiency: how close is grid size to trace count
                    #    Ideal: n_inlines * n_xlines == n_traces (100% coverage, no wasted space)
                    grid_size = estimate['theoretical_traces']
                    size_ratio = min(n_traces / grid_size, grid_size / n_traces) * 100 if grid_size > 0 else 0

                    # Combined score: prioritize unique mapping, then coverage, then size efficiency
                    # Perfect score = 100 + 100 + 100 = 300
                    score = unique_ratio * 1.5 + coverage * 1.0 + size_ratio * 0.5

                    logger.debug(f"  {h1} x {h2}: coverage={coverage:.1f}%, unique={unique_ratio:.1f}%, "
                               f"size_ratio={size_ratio:.1f}%, score={score:.1f}")

                    if score > best_score:
                        best_score = score
                        best_combo = {
                            'inline': h1,
                            'xline': h2,
                            'coverage': coverage,
                            'unique_ratio': unique_ratio,
                            'n_inlines': estimate['n_inlines'],
                            'n_xlines': estimate['n_xlines'],
                            'score': score
                        }
                except Exception as e:
                    logger.debug(f"  {h1} x {h2}: failed - {e}")

        if best_combo:
            logger.info(f"Best combination: {best_combo['inline']} x {best_combo['xline']} "
                       f"({best_combo['n_inlines']}x{best_combo['n_xlines']}) "
                       f"coverage={best_combo['coverage']:.1f}%, unique={best_combo['unique_ratio']:.1f}%")

        return best_combo

    def _on_auto_detect(self):
        """Handle auto-detect button click - find and set best header combination."""
        import logging
        logger = logging.getLogger(__name__)

        # Show busy cursor
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        try:
            best = self._find_best_header_combination()

            if best is None:
                QApplication.restoreOverrideCursor()
                QMessageBox.warning(
                    self,
                    "Auto-detect Failed",
                    "Could not find a suitable header combination.\n"
                    "Please select headers manually."
                )
                return

            logger.info(f"Auto-detected best combination: {best['inline']} x {best['xline']} "
                       f"({best['coverage']:.1f}% coverage)")

            # Find the display items for the best headers
            inline_display = None
            xline_display = None

            for display_item, header in self.display_to_header.items():
                if header == best['inline']:
                    inline_display = display_item
                if header == best['xline']:
                    xline_display = display_item

            if inline_display and xline_display:
                # Block signals to prevent multiple preview updates
                self.inline_combo.blockSignals(True)
                self.xline_combo.blockSignals(True)

                self.inline_combo.setCurrentText(inline_display)
                self.xline_combo.setCurrentText(xline_display)

                self.inline_combo.blockSignals(False)
                self.xline_combo.blockSignals(False)

                # Update preview once
                self._update_preview()

                # Build detailed message
                msg = (
                    f"Best combination found:\n\n"
                    f"Inline: {best['inline']} ({best.get('n_inlines', '?')} values)\n"
                    f"Crossline: {best['xline']} ({best.get('n_xlines', '?')} values)\n\n"
                    f"Grid coverage: {best['coverage']:.1f}%\n"
                    f"Unique mapping: {best.get('unique_ratio', 0):.1f}%"
                )

                QMessageBox.information(
                    self,
                    "Auto-detect Complete",
                    msg
                )
            else:
                QMessageBox.warning(
                    self,
                    "Auto-detect Error",
                    f"Found combination ({best['inline']} x {best['xline']}) "
                    f"but could not set in UI."
                )

        except Exception as e:
            logger.error(f"Auto-detect failed: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"Auto-detect failed:\n{str(e)}"
            )
        finally:
            QApplication.restoreOverrideCursor()

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

    def get_spatial_distances(self) -> Tuple[float, float]:
        """
        Get the spatial distances (dx, dy) from spinboxes.

        Returns:
            Tuple of (dx, dy) in meters
        """
        return (self.dx_spin.value(), self.dy_spin.value())


def build_volume_with_dialog(
    traces_data: np.ndarray,
    headers_df: pd.DataFrame,
    sample_rate_ms: float,
    coordinate_units: str = 'meters',
    coord_scalar: float = 1.0,
    parent=None
) -> Optional[Tuple[SeismicVolume, VolumeGeometry]]:
    """
    Show dialog to select headers and build 3D volume.

    Args:
        traces_data: 2D array of traces (n_samples, n_traces)
        headers_df: DataFrame with trace headers
        sample_rate_ms: Sample rate in milliseconds
        coordinate_units: 'meters' or 'feet'
        coord_scalar: Coordinate scalar from SEG-Y headers
        parent: Parent widget

    Returns:
        Tuple of (SeismicVolume, VolumeGeometry) or None if cancelled
    """
    n_samples = traces_data.shape[0]

    # Show header selection dialog
    dialog = VolumeHeaderDialog(headers_df, n_samples, coord_scalar, parent)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None

    inline_key, xline_key = dialog.get_selected_headers()
    dx, dy = dialog.get_spatial_distances()

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
            dx=dx,
            dy=dy,
            coord_scalar=coord_scalar,
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
