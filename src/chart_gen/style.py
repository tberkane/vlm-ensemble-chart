"""Per-library style randomization."""

import random as _random


# Shared color palettes (hex) usable across all libraries
PALETTE_12 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#393b79", "#637939",
]


def pick_colors(rng: _random.Random, n: int) -> list[str]:
    """Pick *n* distinct hex colors."""
    if n <= len(PALETTE_12):
        return rng.sample(PALETTE_12, n)
    # extend with HSL-spaced hues
    import colorsys
    extra = []
    for i in range(n - len(PALETTE_12)):
        h = (i + 1) / (n - len(PALETTE_12) + 2)
        r, g, b = colorsys.hls_to_rgb(h, 0.5, 0.7)
        extra.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    all_colors = list(PALETTE_12) + extra
    rng.shuffle(all_colors)
    return all_colors[:n]


# ── Matplotlib ──────────────────────────────────────────────────────────────

MPL_STYLES = ["default", "bmh", "classic", "fast", "fivethirtyeight", "ggplot", "grayscale"]
FONTS = ["serif", "sans-serif", "monospace"]
GRID_STYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "v", "<", ">", "D", "P", "*", "X", None]
LINE_STYLES = ["-", "--", "-.", ":", (0, (5, 10)), (0, (1, 1))]
TITLE_WORDS = ["Timeseries", "Trend", "Chart", "Statistics", "Evolution", "Progression", "Data", "World Bank", ""]


def random_matplotlib_style(rng: _random.Random) -> dict:
    """Return a dict consumed by MatplotlibRenderer."""
    return {
        "mpl_styles": rng.sample(MPL_STYLES, k=rng.randint(1, 3)),
        "font_family": rng.choice(FONTS),
        "title_size": rng.choice([14, 16, 18]),
        "label_size": rng.choice([12, 14]),
        "tick_size": rng.choice([10, 12]),
        "grid_on": rng.choice([True, False]),
        "grid_style": rng.choice(GRID_STYLES),
        "grid_alpha": rng.uniform(0.3, 1.0),
        "figsize": (rng.uniform(8, 10), rng.uniform(5.5, 7.5)),
        "title_text": rng.choice(TITLE_WORDS),
    }


def random_line_params(rng: _random.Random) -> dict:
    """Per-series line/marker params for matplotlib/seaborn."""
    marker = rng.choice(MARKERS)
    return {
        "marker": marker if marker is not None and rng.random() < 0.8 else None,
        "linestyle": rng.choice(LINE_STYLES),
        "linewidth": rng.uniform(1, 4),
        "alpha": rng.uniform(0.7, 1.0),
    }


# ── Seaborn ─────────────────────────────────────────────────────────────────

SNS_THEMES = ["whitegrid", "darkgrid", "white", "dark", "ticks"]
SNS_CONTEXTS = ["paper", "notebook", "talk", "poster"]
SNS_PALETTES = ["deep", "muted", "bright", "pastel", "dark", "colorblind", "Set1", "Set2", "husl", "tab10"]


def random_seaborn_style(rng: _random.Random) -> dict:
    return {
        "theme": rng.choice(SNS_THEMES),
        "context": rng.choice(SNS_CONTEXTS),
        "palette": rng.choice(SNS_PALETTES),
        "font_scale": rng.uniform(0.8, 1.4),
        "figsize": (rng.uniform(8, 10), rng.uniform(5.5, 7.5)),
        "title_text": rng.choice(TITLE_WORDS),
        "grid_on": rng.choice([True, False]),
    }


# ── Plotly ──────────────────────────────────────────────────────────────────

PLOTLY_TEMPLATES = ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white"]
PLOTLY_FONTS = ["Arial", "Courier New", "Times New Roman", "Verdana", "Georgia"]


def random_plotly_style(rng: _random.Random) -> dict:
    return {
        "template": rng.choice(PLOTLY_TEMPLATES),
        "font_family": rng.choice(PLOTLY_FONTS),
        "font_size": rng.choice([12, 14, 16]),
        "title_text": rng.choice(TITLE_WORDS),
        "showgrid_x": rng.choice([True, False]),
        "showgrid_y": rng.choice([True, False]),
        "width": rng.randint(800, 1000),
        "height": rng.randint(550, 750),
    }


# ── Bokeh ───────────────────────────────────────────────────────────────────

BOKEH_THEMES = ["caliber", "dark_minimal", "light_minimal", "night_sky", "contrast"]


def random_bokeh_style(rng: _random.Random) -> dict:
    return {
        "theme": rng.choice(BOKEH_THEMES),
        "title_text": rng.choice(TITLE_WORDS),
        "width": rng.randint(800, 1000),
        "height": rng.randint(550, 750),
        "title_font_size": f"{rng.choice([14, 16, 18])}px",
        "axis_label_font_size": f"{rng.choice([12, 14])}px",
        "grid_visible": rng.choice([True, False]),
    }
