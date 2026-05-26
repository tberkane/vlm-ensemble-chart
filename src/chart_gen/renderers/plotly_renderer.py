"""Plotly renderer — static PNG export via kaleido."""

import plotly.graph_objects as go

from .base import ChartRenderer
from ..types import ChartSpec
from ..style import pick_colors
import random as _random_mod


class PlotlyRenderer(ChartRenderer):
    def render(self, spec: ChartSpec, output_path: str) -> None:
        s = spec.style
        rng = _random_mod.Random()
        series_list = self.prepare_series(spec)
        colors = pick_colors(rng, len(series_list))

        dispatch = {
            "line": self._line,
            "area": self._area,
            "grouped_bar": self._grouped_bar,
            "stacked_bar": self._stacked_bar,
        }
        fig = dispatch[spec.chart_type](series_list, colors, rng)

        ylabel = spec.dataset.series_name.replace("_", " ").capitalize()
        fig.update_layout(
            template=s["template"],
            title=s["title_text"] or None,
            xaxis_title="Year",
            yaxis_title=ylabel,
            xaxis=dict(showgrid=s["showgrid_x"], dtick=None),
            yaxis=dict(showgrid=s["showgrid_y"]),
            font=dict(family=s["font_family"], size=s["font_size"]),
            legend=dict(orientation="v"),
            width=s["width"],
            height=s["height"],
        )
        fig.write_image(output_path, format="png", scale=2)

    # ── chart types ─────────────────────────────────────────────────────

    @staticmethod
    def _line(series_list, colors, rng):
        fig = go.Figure()
        for s, color in zip(series_list, colors):
            y = [v for v in s["values"]]
            fig.add_trace(go.Scatter(
                x=s["years"], y=y, mode="lines+markers" if rng.random() < 0.5 else "lines",
                name=s["name"], line=dict(color=color, width=rng.uniform(1.5, 3.5)),
            ))
        return fig

    @staticmethod
    def _area(series_list, colors, rng):
        fig = go.Figure()
        stacked = rng.random() < 0.5
        for i, (s, color) in enumerate(zip(series_list, colors)):
            y = [v if v is not None else 0.0 for v in s["values"]]
            fill = "tonexty" if stacked and i > 0 else "tozeroy"
            fig.add_trace(go.Scatter(
                x=s["years"], y=y, fill=fill, name=s["name"],
                line=dict(color=color, width=rng.uniform(1, 2.5)),
                opacity=rng.uniform(0.5, 0.85),
            ))
        if stacked:
            fig.update_layout(hovermode="x unified")
        return fig

    @staticmethod
    def _grouped_bar(series_list, colors, rng):
        fig = go.Figure()
        for s, color in zip(series_list, colors):
            vals = [v if v is not None else 0.0 for v in s["values"]]
            fig.add_trace(go.Bar(
                x=[str(y) for y in s["years"]], y=vals, name=s["name"],
                marker_color=color, opacity=rng.uniform(0.75, 1.0),
            ))
        fig.update_layout(barmode="group")
        return fig

    @staticmethod
    def _stacked_bar(series_list, colors, rng):
        fig = go.Figure()
        for s, color in zip(series_list, colors):
            vals = [v if v is not None else 0.0 for v in s["values"]]
            fig.add_trace(go.Bar(
                x=[str(y) for y in s["years"]], y=vals, name=s["name"],
                marker_color=color, opacity=rng.uniform(0.75, 1.0),
            ))
        fig.update_layout(barmode="stack")
        return fig
