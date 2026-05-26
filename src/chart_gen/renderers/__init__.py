from .matplotlib_renderer import MatplotlibRenderer
from .seaborn_renderer import SeabornRenderer
from .plotly_renderer import PlotlyRenderer
from .bokeh_renderer import BokehRenderer

RENDERERS = {
    "matplotlib": MatplotlibRenderer,
    "seaborn": SeabornRenderer,
    "plotly": PlotlyRenderer,
    "bokeh": BokehRenderer,
}


def get_renderer(library: str):
    """Return a renderer instance for the given library name."""
    if library not in RENDERERS:
        raise ValueError(f"Unknown library: {library}. Choose from {list(RENDERERS)}")
    return RENDERERS[library]()
