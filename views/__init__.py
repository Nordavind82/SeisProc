"""Views package - UI components and widgets."""
from .seismic_viewer import SeismicViewer
from .seismic_viewer_pyqtgraph import SeismicViewerPyQtGraph
from .control_panel import ControlPanel
from .gather_navigation_panel import GatherNavigationPanel

__all__ = ['SeismicViewer', 'SeismicViewerPyQtGraph', 'ControlPanel', 'GatherNavigationPanel']
