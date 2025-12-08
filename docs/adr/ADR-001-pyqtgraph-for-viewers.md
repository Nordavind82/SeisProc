# ADR-001: PyQtGraph for Seismic Data Visualization

## Status
Accepted

## Date
2024-01-15

## Context
SeisProc requires real-time visualization of seismic data with:
- Interactive zoom, pan, and scroll on datasets with 10K-100K+ traces
- Variable density display with amplitude-to-color mapping
- Synchronized scrolling across multiple viewers
- Sub-second response time for user interactions

We evaluated several visualization libraries:
1. **Matplotlib** - Standard scientific plotting
2. **PyQtGraph** - Qt-native high-performance plotting
3. **VisPy** - GPU-accelerated visualization
4. **Custom OpenGL** - Direct GPU rendering

## Decision
We chose **PyQtGraph** for all seismic data viewers.

## Rationale

### Performance
- PyQtGraph uses NumPy arrays directly, avoiding data copying
- OpenGL acceleration available via `setUseOpenGL(True)`
- ImageItem can display 4000x2000 images at 60fps
- Lazy rendering: only visible portion is rendered

### Qt Integration
- Native Qt widgets integrate seamlessly with PyQt6
- Uses Qt's signal/slot mechanism for event handling
- Consistent look and feel with other Qt widgets
- Easy to embed in splitters, docks, and layouts

### Feature Set
- Built-in `ImageItem` perfect for variable density display
- `ViewBox` provides zoom, pan, scroll out of the box
- `InfiniteLine` and `ROI` for interactive picking
- ColorMap support for seismic-appropriate palettes

### Development Speed
- Fewer lines of code than Matplotlib for equivalent functionality
- Better documentation for real-time applications
- Active community with seismic/scientific users

## Alternatives Considered

### Matplotlib
- **Pros**: Ubiquitous, excellent static plots
- **Cons**: Slow for large datasets, clunky interactivity, not designed for real-time
- **Verdict**: Used for ISA spectrum plots (small, static) but not for main viewers

### VisPy
- **Pros**: Fastest option, GPU-accelerated
- **Cons**: Steeper learning curve, less Qt integration, smaller community
- **Verdict**: Overkill for our needs; PyQtGraph is fast enough

### Custom OpenGL
- **Pros**: Maximum control and performance
- **Cons**: Massive development effort, maintenance burden
- **Verdict**: Not justified for this project scope

## Consequences

### Positive
- 60fps interactive performance on 50K trace gathers
- Rapid development of synchronized viewers
- Easy colormap and gain control implementation

### Negative
- Learning curve for developers unfamiliar with PyQtGraph
- Some advanced plot types require custom implementations
- Documentation can be sparse for edge cases

## References
- PyQtGraph documentation: https://pyqtgraph.readthedocs.io/
- ImageItem performance benchmarks in `tests/benchmarks/`
