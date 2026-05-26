"""Abstract base class for chart renderers."""

from abc import ABC, abstractmethod
from ..types import ChartSpec


class ChartRenderer(ABC):
    """Renders a ChartSpec to a PNG file."""

    @abstractmethod
    def render(self, spec: ChartSpec, output_path: str) -> None:
        """Render the chart described by *spec* and save to *output_path*."""

    # ── helpers shared across renderers ──────────────────────────────────

    @staticmethod
    def prepare_series(spec: ChartSpec) -> list[dict]:
        """Return a list of {name, years, values} dicts, one per country.

        Values are floats or None (for missing data).
        """
        series = []
        for country, data in spec.dataset.countries.items():
            vals = []
            for yr in spec.years:
                v = data.get(yr)
                if v is None or str(v).strip() == "":
                    vals.append(None)
                else:
                    vals.append(float(v))
            series.append({"name": country, "years": list(spec.years), "values": vals})
        return series
