"""Bokeh renderer — static PNG export via selenium + chromedriver."""

from bokeh.plotting import figure
from bokeh.io import export_png
from bokeh.models import ColumnDataSource, FactorRange, Legend
from bokeh.transform import dodge

from .base import ChartRenderer
from ..types import ChartSpec
from ..style import pick_colors
import random as _random_mod

# Ensure chromedriver is available (called once at import time)
try:
    import chromedriver_autoinstaller
    chromedriver_autoinstaller.install()
except Exception:
    pass  # may already be on PATH


class BokehRenderer(ChartRenderer):
    def render(self, spec: ChartSpec, output_path: str) -> None:
        s = spec.style
        rng = _random_mod.Random()
        series_list = self.prepare_series(spec)
        colors = pick_colors(rng, len(series_list))
        ylabel = spec.dataset.series_name.replace("_", " ").capitalize()

        is_bar = spec.chart_type in ("grouped_bar", "stacked_bar")

        # Bar charts need a FactorRange set at construction time
        kwargs = dict(
            width=s["width"],
            height=s["height"],
            title=s["title_text"] if s["title_text"] else None,
            x_axis_label="Year",
            y_axis_label=ylabel,
        )
        if is_bar:
            years_str = [str(y) for y in series_list[0]["years"]]
            kwargs["x_range"] = FactorRange(*years_str)

        p = figure(**kwargs)

        # Title font — only if title exists
        if p.title:
            p.title.text_font_size = s["title_font_size"]
        p.xaxis.axis_label_text_font_size = s["axis_label_font_size"]
        p.yaxis.axis_label_text_font_size = s["axis_label_font_size"]

        if not s["grid_visible"]:
            p.xgrid.visible = False
            p.ygrid.visible = False

        dispatch = {
            "line": self._line,
            "area": self._area,
            "grouped_bar": self._grouped_bar,
            "stacked_bar": self._stacked_bar,
        }
        dispatch[spec.chart_type](p, series_list, colors, rng)

        if is_bar:
            p.xaxis.major_label_orientation = 0.78  # ~45 degrees

        from bokeh.io import curdoc
        curdoc().theme = s.get("theme", "caliber")

        export_png(p, filename=output_path)
        curdoc().theme = "caliber"

    # ── chart types ─────────────────────────────────────────────────────

    @staticmethod
    def _line(p, series_list, colors, rng):
        legend_items = []
        for s, color in zip(series_list, colors):
            y = [v if v is not None else float("nan") for v in s["values"]]
            r = p.line(s["years"], y, line_color=color,
                       line_width=rng.uniform(1.5, 3.5), alpha=rng.uniform(0.7, 1.0))
            legend_items.append((s["name"], [r]))
            if rng.random() < 0.5:
                c = p.scatter(s["years"], y, color=color, size=rng.randint(4, 8))
                legend_items[-1] = (s["name"], [r, c])
        legend = Legend(items=legend_items, location="top_left")
        p.add_layout(legend, "right")

    @staticmethod
    def _area(p, series_list, colors, rng):
        legend_items = []
        stacked = rng.random() < 0.5
        if stacked:
            data = {"years": series_list[0]["years"]}
            names = []
            for s in series_list:
                names.append(s["name"])
                data[s["name"]] = [v if v is not None else 0.0 for v in s["values"]]
            source = ColumnDataSource(data=data)
            renderers = p.varea_stack(
                stackers=names, x="years", source=source,
                color=colors[:len(names)], alpha=rng.uniform(0.5, 0.85),
            )
            for name, r in zip(names, renderers):
                legend_items.append((name, [r]))
        else:
            for s, color in zip(series_list, colors):
                y = [v if v is not None else 0.0 for v in s["values"]]
                r = p.varea(x=s["years"], y1=0, y2=y,
                            fill_color=color, fill_alpha=rng.uniform(0.25, 0.5))
                ln = p.line(s["years"], y, line_color=color, line_width=rng.uniform(1, 2.5))
                legend_items.append((s["name"], [r, ln]))
        legend = Legend(items=legend_items, location="top_left")
        p.add_layout(legend, "right")

    @staticmethod
    def _grouped_bar(p, series_list, colors, rng):
        years_str = [str(y) for y in series_list[0]["years"]]
        n = len(series_list)
        total_width = 0.8
        bar_width = total_width / n
        legend_items = []
        for i, (s, color) in enumerate(zip(series_list, colors)):
            vals = [v if v is not None else 0.0 for v in s["values"]]
            offset = -total_width / 2 + bar_width * (i + 0.5)
            source = ColumnDataSource(data={"years": years_str, "top": vals})
            r = p.vbar(
                x=dodge("years", offset, range=p.x_range),
                top="top", width=bar_width * 0.9,
                source=source,
                color=color, alpha=rng.uniform(0.75, 1.0),
            )
            legend_items.append((s["name"], [r]))
        legend = Legend(items=legend_items, location="top_left")
        p.add_layout(legend, "right")

    @staticmethod
    def _stacked_bar(p, series_list, colors, rng):
        years_str = [str(y) for y in series_list[0]["years"]]
        data = {"years": years_str}
        names = []
        for s in series_list:
            names.append(s["name"])
            data[s["name"]] = [v if v is not None else 0.0 for v in s["values"]]
        source = ColumnDataSource(data=data)
        renderers = p.vbar_stack(
            stackers=names, x="years", width=0.65, source=source,
            color=colors[:len(names)], alpha=rng.uniform(0.75, 1.0),
        )
        legend_items = [(name, [r]) for name, r in zip(names, renderers)]
        legend = Legend(items=legend_items, location="top_left")
        p.add_layout(legend, "right")
