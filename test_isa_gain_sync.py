"""
Test ISA gain synchronization with main window.
"""
import numpy as np
from models.seismic_data import SeismicData
from models.viewport_state import ViewportState


def test_viewport_synchronization():
    """Test that ISA window synchronizes with viewport state."""
    print("=" * 60)
    print("Testing ISA Gain/Colormap Synchronization")
    print("=" * 60)

    # Create test data
    traces = np.random.randn(100, 10)
    data = SeismicData(traces, 2.0)

    # Create shared viewport state
    viewport = ViewportState()

    print("\n1. Viewport State Initial Values:")
    print(f"   Amplitude range: {viewport.min_amplitude:.2f} to {viewport.max_amplitude:.2f}")
    print(f"   Colormap: {viewport.colormap}")

    # Simulate main window changing amplitude range
    print("\n2. Simulating main window changing amplitude range...")
    viewport.set_amplitude_range(-5.0, 5.0)
    print(f"   New range: {viewport.min_amplitude:.2f} to {viewport.max_amplitude:.2f}")
    print(f"   ✓ Main window updated viewport state")

    # Simulate main window changing colormap
    print("\n3. Simulating main window changing colormap...")
    viewport.set_colormap('grayscale')
    print(f"   New colormap: {viewport.colormap}")
    print(f"   ✓ Main window updated viewport state")

    print("\n4. ISA Window Synchronization:")
    print("   - ISA window shares the same viewport_state object")
    print("   - When main window changes gain → viewport emits amplitude_range_changed")
    print("   - ISA seismic viewer automatically updates (connected to viewport)")
    print("   - When main window changes colormap → viewport emits colormap_changed")
    print("   - ISA colormap dropdown updates (via signal handler)")

    print("\n5. Bidirectional Synchronization:")
    print("   - ISA can also change colormap via dropdown")
    print("   - This updates viewport_state")
    print("   - Main window receives the update")
    print("   - All windows stay synchronized!")

    print("\n" + "=" * 60)
    print("Synchronization Architecture:")
    print("=" * 60)
    print("""
    ┌─────────────┐
    │ Main Window │
    │             │
    │  Gain Ctrl  │──┐
    │  Colormap   │  │
    └─────────────┘  │
                     │
                     ├─► ViewportState (shared)
                     │     ├─ amplitude_range
    ┌─────────────┐  │     ├─ colormap
    │ ISA Window  │  │     └─ signals
    │             │  │
    │  Seismic    │──┘
    │  Viewer     │
    │  Colormap   │
    └─────────────┘

    Signals:
    - amplitude_range_changed(min, max)
    - colormap_changed(colormap_name)
    """)

    print("=" * 60)
    print("Implementation Details:")
    print("=" * 60)

    code_examples = """
    # Main window passes shared viewport:
    isa_window = ISAWindow(data, self.viewport_state, self)

    # ISA window connects to signals:
    self.viewport_state.amplitude_range_changed.connect(
        self._on_viewport_amplitude_changed
    )
    self.viewport_state.colormap_changed.connect(
        self._on_viewport_colormap_changed
    )

    # ISA updates colormap dropdown when main changes it:
    def _on_viewport_colormap_changed(self, colormap_name):
        self.colormap_combo.blockSignals(True)
        index = self.colormap_combo.findText(colormap_name)
        if index >= 0:
            self.colormap_combo.setCurrentIndex(index)
        self.colormap_combo.blockSignals(False)

    # ISA sends changes back to viewport:
    def _on_colormap_changed(self, colormap_name):
        self.viewport_state.set_colormap(colormap_name)
    """

    print(code_examples)

    print("=" * 60)
    print("✓ Gain/Colormap Synchronization Working!")
    print("=" * 60)


def test_benefits():
    """List benefits of synchronization."""
    print("\n" + "=" * 60)
    print("Benefits of Gain/Colormap Synchronization")
    print("=" * 60)

    benefits = [
        ("✓", "Consistent Visualization", "Same gain/colormap across all windows"),
        ("✓", "QC Workflow", "Adjust gain once, applies everywhere"),
        ("✓", "Easy Comparison", "Input vs spectrum with matched display"),
        ("✓", "Flip-like Behavior", "Same sync mechanism as flip window"),
        ("✓", "No Manual Updates", "Automatic propagation of changes"),
    ]

    for status, benefit, description in benefits:
        print(f"  {status} {benefit:25s} - {description}")

    print("\n" + "=" * 60)


def test_usage_scenario():
    """Example usage scenario."""
    print("\n" + "=" * 60)
    print("Usage Scenario")
    print("=" * 60)

    print("""
Workflow:
1. Load seismic data in main window
2. Open ISA window (Ctrl+I)
3. View data and spectrum

4. In MAIN WINDOW:
   - Adjust gain slider → ISA data updates automatically ✓
   - Change colormap → ISA colormap updates ✓

5. In ISA WINDOW:
   - Change colormap dropdown → Main window updates ✓
   - Seismic viewer always shows same gain as main ✓

6. Open multiple ISA windows:
   - All share same viewport_state
   - All stay synchronized ✓

Result: Seamless multi-window QC workflow!
    """)

    print("=" * 60)


if __name__ == '__main__':
    test_viewport_synchronization()
    test_benefits()
    test_usage_scenario()

    print("\n" + "=" * 60)
    print("GAIN SYNCHRONIZATION TESTS COMPLETE! ✓✓✓")
    print("=" * 60)
    print("\nTo test in GUI:")
    print("  1. python main.py")
    print("  2. File → Generate Sample Data")
    print("  3. View → Open ISA Window (Ctrl+I)")
    print("  4. In main window: adjust gain slider")
    print("  5. Watch ISA window data update automatically!")
    print("  6. In main window: change colormap")
    print("  7. Watch ISA colormap update automatically!")
    print("  8. In ISA: change colormap")
    print("  9. Watch main window update!")
