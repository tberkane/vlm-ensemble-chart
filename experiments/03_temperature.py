import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

# =========================
# ACL sizing (two columns for side-by-side panels)
# =========================
ACL_COL_WIDTH_IN = 3.25  # typical \columnwidth in ACL templates
ACL_ASPECT = 0.70  # height = width * aspect; tweak 0.60–0.85 as needed
FIG_W = ACL_COL_WIDTH_IN  # Double width for two panels
FIG_H = ACL_COL_WIDTH_IN * ACL_ASPECT * 2

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
        # For a 3.25in wide figure, smaller fonts read better in 2-col layouts
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

TEMP0_SUMMARY = Path(
    "outputs/experiments/2025-12-11_140817__ensemble_size_full_100_7runs_temp0/summary_results.json"
)
TEMP1_SUMMARY = Path(
    "outputs/experiments/2025-12-11_140919__ensemble_size_full_100_7runs_temp1/summary_results.json"
)
TEMP2_SUMMARY = Path(
    "outputs/experiments/2025-12-11_142221__ensemble_size_full_100_7runs_temp2/summary_results.json"
)
OUTPUT_PLOT = Path("outputs/experiments/temperature_ablation.png")


def load_series(summary_path: Path):
    with summary_path.open() as f:
        data = json.load(f)

    medoid_results = data["results"]["medoid"]
    rows = sorted(((int(k), v) for k, v in medoid_results.items()), key=lambda t: t[0])

    ensemble_sizes = [k for k, _ in rows]
    mean_f1 = [r["mean_f1"] for _, r in rows]
    ci_lower = [r["ci_lower"] for _, r in rows]
    ci_upper = [r["ci_upper"] for _, r in rows]
    return ensemble_sizes, mean_f1, ci_lower, ci_upper


# WB-ChartExtract data
WB_TEMP0_F1 = [
    47.07,
    47.47,
    49.28,
    49.62,
    50.16,
    50.23,
    50.34,
    50.55,
    50.60,
    50.70,
    50.66,
    50.76,
    50.74,
    50.81,
    50.92,
    51.01,
    50.94,
    51.04,
    51.03,
    51.05,
]
WB_TEMP1_F1 = [
    47.12,
    47.66,
    49.87,
    50.09,
    50.38,
    50.55,
    50.45,
    50.8,
    50.79,
    50.95,
    50.86,
    50.99,
    50.92,
    50.89,
    51.67,
    51.71,
    51.68,
    51.7,
    51.65,
    51.72,
]

WB_TEMP2_F1 = [
    45.00,
    41.95,
    47.07,
    47.14,
    48.82,
    50.11,
    50.21,
    50.45,
    50.50,
    50.55,
    50.56,
    50.56,
    50.67,
    50.78,
    50.78,
    51.00,
    50.85,
    51.04,
    51.0,
    51.01,
]


def plot_temperatures():
    plt.style.use("default")

    # Use constrained_layout for stable sizing (avoid bbox_inches="tight" shrink-wrapping)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(FIG_W, FIG_H), constrained_layout=True
    )

    colors = ["#648fff", "#ffb000", "#dc267f"]
    linestyles = ["-", "--", "-."]
    markers = ["o", "s", "^"]

    # Panel 1: ChartQA Dataset
    for summary_path, label, color, linestyle, marker in [
        (TEMP0_SUMMARY, r"$T=0.0$", colors[0], linestyles[0], markers[0]),
        (TEMP1_SUMMARY, r"$T=1.0$", colors[1], linestyles[1], markers[1]),
        (TEMP2_SUMMARY, r"$T=2.0$", colors[2], linestyles[2], markers[2]),
    ]:
        sizes, means, ci_low, ci_up = load_series(summary_path)
        ax1.plot(
            sizes,
            means,
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.2,
            markersize=4.0,
            markeredgewidth=0.4,
            markeredgecolor="white",
        )
        ax1.fill_between(sizes, ci_low, ci_up, color=color, alpha=0.15, linewidth=0)

    ax1.set_xlabel("Ensemble Size")
    ax1.set_ylabel("F1")
    ax1.set_title("ChartQA", fontsize=9, fontweight="bold")

    ax1.set_xticks(range(1, 8))
    ax1.set_xlim(0.5, 7.5)

    ax1.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax1.set_axisbelow(True)

    ax1.legend(
        loc="best",
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        borderpad=0.3,
        handlelength=2.0,
        handletextpad=0.6,
        labelspacing=0.3,
    )

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_linewidth(0.6)
    ax1.spines["bottom"].set_linewidth(0.6)

    # Panel 2: WB-ChartExtract Dataset
    for f1_values, label, color, linestyle, marker in [
        (WB_TEMP0_F1, r"$T=0.0$", colors[0], linestyles[0], markers[0]),
        (WB_TEMP1_F1, r"$T=1.0$", colors[1], linestyles[1], markers[1]),
        (WB_TEMP2_F1, r"$T=2.0$", colors[2], linestyles[2], markers[2]),
    ]:
        iterations = list(range(1, len(f1_values) + 1))
        ax2.plot(
            iterations,
            f1_values,
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.2,
            markersize=4.0,
            markeredgewidth=0.4,
            markeredgecolor="white",
        )

    ax2.set_xlabel("Ensemble Size")
    ax2.set_ylabel("F1")
    ax2.set_title("WB-ChartExtract", fontsize=9, fontweight="bold")

    ax2.set_xticks(range(1, 16))
    # Show only every other x-axis label
    labels = [str(i) if i % 2 == 1 else "" for i in range(1, 16)]
    ax2.set_xticklabels(labels)
    ax2.set_xlim(0.5, 15.5)

    ax2.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax2.set_axisbelow(True)

    ax2.legend(
        loc="best",
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        borderpad=0.3,
        handlelength=2.0,
        handletextpad=0.6,
        labelspacing=0.3,
    )

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_linewidth(0.6)
    ax2.spines["bottom"].set_linewidth(0.6)

    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: omit bbox_inches="tight" to preserve the intended physical size
    fig.savefig(OUTPUT_PLOT, dpi=600, facecolor="white")
    pdf_path = OUTPUT_PLOT.with_suffix(".pdf")
    fig.savefig(pdf_path, dpi=600, facecolor="white")
    print(f"Saved plot to {OUTPUT_PLOT}")
    print(f"Saved plot to {pdf_path}")


if __name__ == "__main__":
    plot_temperatures()
