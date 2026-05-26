from dataclasses import dataclass, field


CHART_TYPES = ["line", "area", "grouped_bar", "stacked_bar"]
LIBRARIES = ["matplotlib", "seaborn", "plotly", "bokeh"]


@dataclass
class Dataset:
    """One chart's underlying data: 1 indicator × 2-3 countries over time."""
    index: int  # 1-based chart ID
    series_name: str
    countries: dict  # {country_name: {year(int): value(float|None)}}


@dataclass
class ChartSpec:
    """Full specification for rendering one chart."""
    dataset: Dataset
    chart_type: str  # one of CHART_TYPES
    library: str  # one of LIBRARIES
    years: list  # actual years to plot (may be subsampled for bar charts)
    style: dict = field(default_factory=dict)  # library-specific style params


@dataclass
class ChartMetadata:
    """Saved alongside each chart for downstream analysis."""
    index: int
    chart_type: str
    library: str
    series_name: str
    countries: list
    num_years: int
    subsampled: bool
