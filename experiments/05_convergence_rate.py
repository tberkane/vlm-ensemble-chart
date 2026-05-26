from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

# =========================
# ACL sizing (two columns for side-by-side panels)
# =========================
ACL_COL_WIDTH_IN = 3.25  # typical \columnwidth in ACL templates
ACL_ASPECT = 0.55  # height = width * aspect; tweak 0.60–0.85 as needed
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


OUTPUT_PLOT = Path("outputs/experiments/convergence_rate_comparison.png")

# ChartQA data
CHARTQA_F1 = [
    80.90,
    81.52,
    85.01,
    84.96,
    85.92,
    85.89,
    86.14,
    86.15,
    86.16,
    86.31,
    86.26,
    86.35,
    86.36,
    86.37,
    86.35,
    86.38,
    86.38,
    86.38,
    86.38,
    86.39,
]

CHARTQA_CONVERGENCE = [
    0.0,
    0.0,
    48.2,
    71.4,
    85.8,
    88.8,
    92.8,
    93.9,
    95.4,
    96.0,
    96.9,
    97.2,
    97.7,
    97.7,
    98.1,
    98.2,
    98.2,
    98.3,
    98.5,
    98.5,
]

# WB-ChartExtract data
WB_F1 = [
    47.12,
    47.66,
    49.87,
    49.97,
    50.32,
    50.43,
    50.34,
    50.51,
    50.44,
    50.50,
    50.51,
    50.51,
    50.52,
    50.53,
    50.88,
    50.87,
    50.85,
    50.84,
    50.85,
    50.85,
]

WB_CONVERGENCE = [
    0.0,
    0.0,
    14.6,
    22.4,
    38.1,
    44.7,
    54.8,
    58.3,
    64.7,
    68.2,
    72.1,
    74.5,
    77.4,
    78.6,
    80.1,
    80.6,
    84.3,
    85.1,
    87.8,
    88.7,
]


def find_median_stopping_point(convergence_rates):
    """Find the ensemble size where convergence rate first reaches or exceeds 50%."""
    for i, rate in enumerate(convergence_rates):
        if rate >= 50.0:
            return i + 1  # ensemble sizes are 1-indexed
    return None


def plot_convergence_rate():
    plt.style.use("default")

    # Use constrained_layout for stable sizing (avoid bbox_inches="tight" shrink-wrapping)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(FIG_W, FIG_H), constrained_layout=True, sharex=True
    )

    # Colors for F1 and convergence rate
    f1_color = "#648fff"
    convergence_color = "#dc267f"

    # Panel 1: ChartQA Dataset
    ensemble_sizes_chartqa = list(range(1, len(CHARTQA_F1) + 1))

    # Plot F1 on left y-axis
    ax1_f1 = ax1
    ax1_f1.plot(
        ensemble_sizes_chartqa,
        CHARTQA_F1,
        label="F1",
        color=f1_color,
        marker="o",
        linestyle="-",
        linewidth=1.2,
        markersize=4.0,
        markeredgewidth=0.4,
        markeredgecolor="white",
    )
    ax1_f1.set_ylabel("F1", color=f1_color)
    ax1_f1.tick_params(axis="y", labelcolor=f1_color)

    # Plot convergence rate on right y-axis
    ax1_conv = ax1.twinx()
    ax1_conv.plot(
        ensemble_sizes_chartqa,
        CHARTQA_CONVERGENCE,
        label="Convergence Rate",
        color=convergence_color,
        marker="s",
        linestyle="--",
        linewidth=1.2,
        markersize=4.0,
        markeredgewidth=0.4,
        markeredgecolor="white",
    )
    ax1_conv.set_ylabel("Convergence Rate (%)", color=convergence_color)
    ax1_conv.tick_params(axis="y", labelcolor=convergence_color)
    ax1_conv.set_ylim(0, 100)

    ax1.set_title("ChartQA", fontsize=9, fontweight="bold")
    ax1.set_xticks(range(1, len(CHARTQA_F1) + 1))
    # Show only every other x-axis label
    labels = [str(i) if i % 2 == 1 else "" for i in range(1, len(CHARTQA_F1) + 1)]
    ax1.set_xticklabels(labels)
    ax1.set_xlim(0.5, len(CHARTQA_F1) + 0.5)

    ax1.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax1.set_axisbelow(True)

    # Combine legends from both axes
    # lines1, labels1 = ax1_f1.get_legend_handles_labels()
    # lines2, labels2 = ax1_conv.get_legend_handles_labels()
    # ax1.legend(
    #     lines1 + lines2,
    #     labels1 + labels2,
    #     loc="best",
    #     frameon=True,
    #     fancybox=False,
    #     framealpha=0.95,
    #     borderpad=0.3,
    #     handlelength=2.0,
    #     handletextpad=0.6,
    #     labelspacing=0.3,
    # )

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_linewidth(0.6)
    ax1.spines["bottom"].set_linewidth(0.6)

    # Add vertical line for median stopping point
    median_stop_chartqa = find_median_stopping_point(CHARTQA_CONVERGENCE)
    if median_stop_chartqa is not None:
        y_max_f1 = max(CHARTQA_F1)
        y_max_conv = max(CHARTQA_CONVERGENCE)
        ax1.axvline(
            x=median_stop_chartqa,
            color="gray",
            linestyle=":",
            linewidth=1.0,
            alpha=0.7,
        )
        ax1.text(
            median_stop_chartqa,
            y_max_f1 * 0.95,
            f"Median\nstop",
            ha="center",
            va="top",
            fontsize=8,
            color="gray",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.8,
            ),
        )

    # Panel 2: WB-ChartExtract Dataset
    ensemble_sizes_wb = list(range(1, len(WB_F1) + 1))

    # Plot F1 on left y-axis
    ax2_f1 = ax2
    ax2_f1.plot(
        ensemble_sizes_wb,
        WB_F1,
        label="F1",
        color=f1_color,
        marker="o",
        linestyle="-",
        linewidth=1.2,
        markersize=4.0,
        markeredgewidth=0.4,
        markeredgecolor="white",
    )
    ax2_f1.set_xlabel("Ensemble Size")
    ax2_f1.set_ylabel("F1", color=f1_color)
    ax2_f1.tick_params(axis="y", labelcolor=f1_color)

    # Plot convergence rate on right y-axis
    ax2_conv = ax2.twinx()
    ax2_conv.plot(
        ensemble_sizes_wb,
        WB_CONVERGENCE,
        label="Convergence Rate",
        color=convergence_color,
        marker="s",
        linestyle="--",
        linewidth=1.2,
        markersize=4.0,
        markeredgewidth=0.4,
        markeredgecolor="white",
    )
    ax2_conv.set_ylabel("Convergence Rate (%)", color=convergence_color)
    ax2_conv.tick_params(axis="y", labelcolor=convergence_color)
    ax2_conv.set_ylim(0, 100)

    ax2.set_title("WB-ChartExtract", fontsize=9, fontweight="bold")
    ax2.set_xticks(range(1, len(WB_F1) + 1))
    # Show only every other x-axis label
    labels = [str(i) if i % 2 == 1 else "" for i in range(1, len(WB_F1) + 1)]
    ax2.set_xticklabels(labels)
    ax2.set_xlim(0.5, len(WB_F1) + 0.5)

    ax2.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax2.set_axisbelow(True)

    # Combine legends from both axes
    # lines1, labels1 = ax2_f1.get_legend_handles_labels()
    # lines2, labels2 = ax2_conv.get_legend_handles_labels()
    # ax2.legend(
    #     lines1 + lines2,
    #     labels1 + labels2,
    #     loc="best",
    #     frameon=True,
    #     fancybox=False,
    #     framealpha=0.95,
    #     borderpad=0.3,
    #     handlelength=2.0,
    #     handletextpad=0.6,
    #     labelspacing=0.3,
    # )

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_linewidth(0.6)
    ax2.spines["bottom"].set_linewidth(0.6)

    # Add vertical line for median stopping point
    median_stop_wb = find_median_stopping_point(WB_CONVERGENCE)
    if median_stop_wb is not None:
        y_max_f1 = max(WB_F1)
        y_max_conv = max(WB_CONVERGENCE)
        ax2.axvline(
            x=median_stop_wb,
            color="gray",
            linestyle=":",
            linewidth=1.0,
            alpha=0.7,
        )
        ax2.text(
            median_stop_wb,
            y_max_f1 * 0.95,
            f"Median\nstop",
            ha="center",
            va="top",
            fontsize=8,
            color="gray",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.8,
            ),
        )

    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: omit bbox_inches="tight" to preserve the intended physical size
    fig.savefig(OUTPUT_PLOT, dpi=600, facecolor="white")
    pdf_path = OUTPUT_PLOT.with_suffix(".pdf")
    fig.savefig(pdf_path, dpi=600, facecolor="white")
    print(f"Saved plot to {OUTPUT_PLOT}")
    print(f"Saved plot to {pdf_path}")


if __name__ == "__main__":
    plot_convergence_rate()
