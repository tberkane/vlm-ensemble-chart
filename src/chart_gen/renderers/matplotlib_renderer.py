"""Matplotlib renderer — line, area, grouped bar, stacked bar."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from .base import ChartRenderer
from ..types import ChartSpec
from ..style import pick_colors, random_line_params
import random as _random_mod


class MatplotlibRenderer(ChartRenderer):
    def render(self, spec: ChartSpec, output_path: str) -> None:
        s = spec.style
        rng = _random_mod.Random()

        with plt.style.context(s["mpl_styles"]):
            plt.rcParams.update({
                "font.family": s["font_family"],
                "axes.titlesize": s["title_size"],
                "axes.labelsize": s["label_size"],
                "xtick.labelsize": s["tick_size"],
                "ytick.labelsize": s["tick_size"],
                "axes.grid": s["grid_on"],
                "grid.linestyle": s["grid_style"],
                "grid.alpha": s["grid_alpha"],
            })

            fig, ax = plt.subplots(figsize=s["figsize"])
            series_list = self.prepare_series(spec)
            colors = pick_colors(rng, len(series_list))

            dispatch = {
                "line": self._line,
                "area": self._area,
                "grouped_bar": self._grouped_bar,
                "stacked_bar": self._stacked_bar,
            }
            dispatch[spec.chart_type](ax, series_list, colors, rng)

            # Formatting: integer years on x-axis; smart formatting on y-axis
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
            yfmt = mticker.ScalarFormatter(useOffset=False)
            yfmt.set_scientific(False)
            ax.yaxis.set_major_formatter(yfmt)
            if s["title_text"]:
                ax.set_title(s["title_text"], fontsize=s["title_size"])
            ax.set_xlabel("Year", fontsize=s["label_size"])
            ax.set_ylabel(
                spec.dataset.series_name.replace("_", " ").capitalize(), fontsize=11
            )
            if spec.chart_type in ("grouped_bar", "stacked_bar"):
                plt.xticks(rotation=45, ha="right")
            ax.legend(loc="best", fontsize=rng.choice([10, 11, 12]))
            plt.tight_layout()
            plt.savefig(output_path, dpi=100)
            plt.close(fig)

    # ── chart type implementations ──────────────────────────────────────

    @staticmethod
    def _line(ax, series_list, colors, rng):
        for s, color in zip(series_list, colors):
            p = random_line_params(rng)
            y = [v if v is not None else np.nan for v in s["values"]]
            ax.plot(
                s["years"], y, label=s["name"], color=color,
                marker=p["marker"], linestyle=p["linestyle"],
                linewidth=p["linewidth"], alpha=p["alpha"],
            )

    @staticmethod
    def _area(ax, series_list, colors, rng):
        stacked = rng.random() < 0.5
        if stacked:
            years = series_list[0]["years"]
            labels = [s["name"] for s in series_list]
            data = np.array([
                [v if v is not None else 0.0 for v in s["values"]]
                for s in series_list
            ])
            ax.stackplot(years, data, labels=labels, colors=colors,
                         alpha=rng.uniform(0.5, 0.85))
        else:
            for s, color in zip(series_list, colors):
                y = np.array([v if v is not None else np.nan for v in s["values"]])
                ax.fill_between(s["years"], y, alpha=rng.uniform(0.25, 0.5),
                                color=color, label=s["name"])
                ax.plot(s["years"], y, color=color, linewidth=rng.uniform(1, 2.5))

    @staticmethod
    def _grouped_bar(ax, series_list, colors, rng):
        years = np.array(series_list[0]["years"], dtype=float)
        n = len(series_list)
        spacing = years[1] - years[0] if len(years) > 1 else 1
        width = spacing * 0.8 / n
        for i, (s, color) in enumerate(zip(series_list, colors)):
            vals = [v if v is not None else 0.0 for v in s["values"]]
            offset = -spacing * 0.4 + width * (i + 0.5)
            ax.bar(years + offset, vals, width,
                   label=s["name"], color=color, alpha=rng.uniform(0.75, 1.0),
                   edgecolor=rng.choice(["none", "black", "white"]))
        ax.set_xticks(years)
        ax.set_xticklabels([str(int(y)) for y in years])

    @staticmethod
    def _stacked_bar(ax, series_list, colors, rng):
        years = np.array(series_list[0]["years"], dtype=float)
        spacing = years[1] - years[0] if len(years) > 1 else 1
        bar_width = spacing * rng.uniform(0.5, 0.8)
        bottom = np.zeros(len(years))
        for s, color in zip(series_list, colors):
            vals = np.array([v if v is not None else 0.0 for v in s["values"]])
            ax.bar(years, vals, bar_width, bottom=bottom, label=s["name"],
                   color=color, alpha=rng.uniform(0.75, 1.0),
                   edgecolor=rng.choice(["none", "black", "white"]))
            bottom += vals
        ax.set_xticks(years)
        ax.set_xticklabels([str(int(y)) for y in years])
