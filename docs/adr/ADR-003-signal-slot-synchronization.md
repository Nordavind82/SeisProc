# ADR-003: Qt Signal/Slot for State Synchronization

## Status
Accepted

## Date
2024-02-01

## Context
SeisProc has multiple UI components that must stay synchronized:
- Three seismic viewers (Input, Processed, Difference)
- Control panel (gain, colormap, frequency settings)
- Navigation panel (gather selection, sorting)
- Auxiliary windows (Flip, ISA, FK Designer)

Changes in one component must propagate to others:
- Zoom in one viewer → all viewers zoom
- Change colormap → all viewers update
- Navigate to new gather → all viewers reload

Options considered:
1. **Qt Signal/Slot** - Native Qt mechanism
2. **Observer Pattern** - Manual implementation
3. **Redux-like Store** - Centralized state
4. **Shared Mutable State** - Direct references

## Decision
Use **Qt Signal/Slot** via a centralized **ViewportState** object.

## Rationale

### Qt Signal/Slot Benefits

#### Thread Safety
- Signals queued across threads automatically
- No manual locking for UI updates
- Safe for background processing callbacks

#### Loose Coupling
- Emitter doesn't know about receivers
- Add/remove listeners dynamically
- Easy to test components in isolation

#### Qt Integration
- Native to PyQt6, no additional dependencies
- Works with Qt event loop
- Debuggable with Qt tools

### ViewportState Pattern

We created a `ViewportState` class that:
1. Holds shared state (zoom, pan, amplitude range, colormap)
2. Emits signals when state changes
3. Is passed to all components that need synchronization

```python
class ViewportState(QObject):
    # Signals
    time_range_changed = pyqtSignal(float, float)
    trace_range_changed = pyqtSignal(float, float)
    amplitude_range_changed = pyqtSignal(float, float)
    colormap_changed = pyqtSignal(str)

    def set_time_range(self, t_min, t_max):
        if (t_min, t_max) != (self._t_min, self._t_max):
            self._t_min, self._t_max = t_min, t_max
            self.time_range_changed.emit(t_min, t_max)
```

### Connection Example

```python
# In MainWindow
self.viewport_state = ViewportState()

# All viewers share the same state
self.input_viewer = SeismicViewer("Input", self.viewport_state)
self.processed_viewer = SeismicViewer("Processed", self.viewport_state)

# Control panel connects to state
self.control_panel.colormap_combo.currentTextChanged.connect(
    self.viewport_state.set_colormap
)
```

## Alternatives Considered

### Manual Observer Pattern
- **Pros**: No Qt dependency for logic
- **Cons**: Reinventing the wheel, error-prone
- **Verdict**: Qt signals are better tested and maintained

### Redux-like Store
- **Pros**: Predictable state, time-travel debugging
- **Cons**: Overkill for this app size, non-Pythonic
- **Verdict**: Would add complexity without clear benefit

### Shared Mutable State
- **Pros**: Simple to understand
- **Cons**: Race conditions, no change notifications, tight coupling
- **Verdict**: Doesn't scale, hard to debug

## Consequences

### Positive
- Clean separation between components
- Easy to add new synchronized components
- State changes are explicit and traceable
- Thread-safe by default

### Negative
- Signal overhead for high-frequency updates (mitigated by throttling)
- Debugging signal chains can be tricky
- Must remember to disconnect signals to avoid memory leaks

### Guidelines

1. **One source of truth**: ViewportState owns the canonical state
2. **Emit on change**: Only emit if value actually changed
3. **Block signals when batch updating**: Use `blockSignals()` for bulk changes
4. **Disconnect on destroy**: Clean up in `closeEvent` or `__del__`

## References
- Qt Signal/Slot docs: https://doc.qt.io/qt-6/signalsandslots.html
- `models/viewport_state.py` implementation
- PyQt6 best practices: https://www.riverbankcomputing.com/static/Docs/PyQt6/
