import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# =========================
# ACL sizing (single column)
# =========================
ACL_COL_WIDTH_IN = 3.25  # typical \columnwidth in ACL templates
ACL_ASPECT = 0.70
FIG_W = ACL_COL_WIDTH_IN
FIG_H = ACL_COL_WIDTH_IN * ACL_ASPECT * 1.3

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Roboto",
            "DejaVu Sans",
            "Arial",
            "Helvetica",
            "sans-serif",
        ],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 9,
        "text.usetex": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

OUTPUT_PLOT = Path("outputs/experiments/error_analysis.png")

# WB-ChartExtract v2 error breakdown: [correct, value_errors, label_errors, missing, extra]
# Computed from outputs/{predict,adaptive}/... via scripts/compute_error_breakdown.py
DATA = {
    "TinyChart": {
        "before": [28.61, 24.89, 1.21, 32.63, 12.66],
        "after": [29.37, 24.15, 1.26, 32.68, 12.54],
    },
    "Scout": {
        "before": [30.41, 37.70, 1.41, 28.25, 2.23],
        "after": [33.14, 39.55, 0.50, 21.82, 4.99],
    },
    "Qwen3-VL": {
        "before": [52.91, 21.96, 0.13, 24.41, 0.59],
        "after": [56.23, 20.41, 0.16, 22.87, 0.33],
    },
    "Seed": {
        "before": [35.08, 22.11, 0.37, 41.69, 0.75],
        "after": [43.17, 18.97, 0.13, 37.19, 0.54],
    },
    # GPT-5.1 is evaluated single-pass only (no ensembling); shown as one bar.
    "GPT-5.1": {
        "single": [51.26, 22.78, 0.15, 15.70, 10.12],
    },
}

COMPONENTS = ["correct", "value_errors", "label_errors", "missing", "extra"]
COMPONENT_LABELS = ["Correct", "Value Errors", "Label Errors", "Missing", "Extra"]
# Color-blind friendly palette (see: Datawrapper palette "Color Blind Safe")
COMPONENT_COLORS = [
    "#648fff",  # correct
    "#785ef0",  # value_errors
    "#fe6100",  # label_errors
    "#dc267f",  # missing
    "#ffb000",  # extra
]


def plot_error_analysis():
    plt.style.use("default")

    fig, ax = plt.subplots(1, 1, figsize=(FIG_W * 2, FIG_H), constrained_layout=True)

    models = list(DATA.keys())
    n_models = len(models)
    bar_width = 0.35
    x = np.arange(n_models)

    before_bottom = np.zeros(n_models)
    after_bottom = np.zeros(n_models)
    single_bottom = np.zeros(n_models)

    for i, component in enumerate(COMPONENTS):
        for m, model in enumerate(models):
            entry = DATA[model]
            if "single" in entry:
                # Single-pass-only model (e.g. GPT-5.1): one centered bar.
                val = entry["single"][i]
                ax.bar(
                    x[m],
                    val,
                    bar_width,
                    color=COMPONENT_COLORS[i],
                    bottom=single_bottom[m],
                    edgecolor="white",
                    linewidth=0.5,
                )
                single_bottom[m] += val
            else:
                bval = entry["before"][i]
                aval = entry["after"][i]
                ax.bar(
                    x[m] - bar_width / 2,
                    bval,
                    bar_width,
                    color=COMPONENT_COLORS[i],
                    bottom=before_bottom[m],
                    edgecolor="white",
                    linewidth=0.5,
                    hatch="///",
                )
                ax.bar(
                    x[m] + bar_width / 2,
                    aval,
                    bar_width,
                    color=COMPONENT_COLORS[i],
                    bottom=after_bottom[m],
                    edgecolor="white",
                    linewidth=0.5,
                )
                before_bottom[m] += bval
                after_bottom[m] += aval

    ax.set_xlabel("Model", fontsize=13)
    ax.set_ylabel("Percentage (%)", fontsize=13)
    ax.set_xticks(x)
    xlabels = [
        f"{m}\n(single-pass)" if "single" in DATA[m] else m for m in models
    ]
    ax.set_xticklabels(xlabels, rotation=0, ha="center")
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5, axis="y")
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)

    # Make room for legend above
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.85))

    before_patch = Patch(
        facecolor="gray", edgecolor="black", linewidth=0.5, hatch="///"
    )
    after_patch = Patch(facecolor="gray", edgecolor="black", linewidth=0.5)
    fig.legend(
        [before_patch, after_patch] + [Patch(facecolor=c) for c in COMPONENT_COLORS],
        ["Before", "After"] + COMPONENT_LABELS,
        loc="upper center",
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        borderpad=0.3,
        handlelength=1.0,
        handletextpad=0.5,
        labelspacing=0.3,
        fontsize=11,
        ncol=4,
        bbox_to_anchor=(0.5, 0.995),
        bbox_transform=fig.transFigure,
    )

    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(OUTPUT_PLOT, dpi=600, facecolor="white")
    pdf_path = OUTPUT_PLOT.with_suffix(".pdf")
    fig.savefig(pdf_path, dpi=600, facecolor="white")
    print(f"Saved plot to {OUTPUT_PLOT}")
    print(f"Saved plot to {pdf_path}")


if __name__ == "__main__":
    plot_error_analysis()
