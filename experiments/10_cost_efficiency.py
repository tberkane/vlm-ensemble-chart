"""Cost-efficiency scatter: per-image API cost (log scale) vs RMS-F1.

Two panels: ChartQA (left), WB-ChartExtract (right). Single-pass models as
circles; self-ensembled variants as diamonds. Arrows connect each base model
to its ensembled variant.

Numbers come from the paper's Tables 1 and 2 (data as of 2026-05-04).
TinyChart and OneChart are free and cannot be placed on a log scale, so we
omit them with a note in the caption.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ACL sizing - two-panel figure, full text width
FIG_W = 7.0
FIG_H = 3.4

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Roboto", "DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
    "text.usetex": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

OUTPUT = Path("outputs/experiments/cost_efficiency.png")

# (cost_USD, n_charts) -> per-chart cost in millicents
def mc(cost_usd, n):
    return cost_usd * 100_000.0 / n


N_CQ = 1509
N_WB = 1000

# Per panel: model -> (color, single (cost, f1), ensemble (cost, f1) or None,
#                       sp_offset, ens_offset)
# offsets are (x_mult, y_offset) for label placement
CQ = {
    "Qwen3-VL":          ("#2ca02c", (mc(0.42, N_CQ), 91.43), (mc(1.78, N_CQ), 93.21),
                          ("below", 0), ("above", 0)),
    "Scout":             ("#1f77b4", (mc(0.24, N_CQ), 75.13), (mc(2.29, N_CQ), 77.10),
                          ("below", 0), ("right", 0)),
    "Seed":              ("#ff7f0e", (mc(0.76, N_CQ), 81.62), (mc(5.30, N_CQ), 87.04),
                          ("below", 0), ("left", 0)),
    "GPT-5.1":           ("#d62728", (mc(2.93, N_CQ), 84.80), None,
                          ("right", 0), None),
    "Claude Opus 4.6":   ("#9467bd", (mc(32.13, N_CQ), 87.71), None,
                          ("below", 0), None),
    "Gemini 2.5 Pro":    ("#e377c2", (mc(7.04, N_CQ), 88.23), None,
                          ("above", 0), None),
}

WB = {
    "Qwen3-VL":          ("#2ca02c", (mc(0.63, N_WB), 52.91), (mc(9.03, N_WB), 56.23),
                          ("below", 0), ("above", 0)),
    "Scout":             ("#1f77b4", (mc(0.28, N_WB), 30.41), (mc(4.97, N_WB), 33.14),
                          ("below", 0), ("above", 0)),
    "Seed":              ("#ff7f0e", (mc(1.06, N_WB), 35.08), (mc(14.63, N_WB), 43.17),
                          ("below", 0), ("above", 0)),
    "GPT-5.1":           ("#d62728", (mc(4.55, N_WB), 51.26), None,
                          ("left", 0), None),
    "Claude Opus 4.6":   ("#9467bd", (mc(39.17, N_WB), 60.99), None,
                          ("above", 0), None),
    "Gemini 2.5 Pro":    ("#e377c2", (mc(11.09, N_WB), 87.83), None,
                          ("below", 0), None),
}


def _place(ax, x, y, text, color, position, fontsize=7.5):
    """Place a label using a coarse position spec."""
    if position == "right":
        ax.text(x * 1.18, y, text, fontsize=fontsize, color=color,
                va="center", ha="left")
    elif position == "left":
        ax.text(x / 1.18, y, text, fontsize=fontsize, color=color,
                va="center", ha="right")
    elif position == "above":
        ax.text(x, y + 1.6, text, fontsize=fontsize, color=color,
                va="bottom", ha="center")
    elif position == "below":
        ax.text(x, y - 1.6, text, fontsize=fontsize, color=color,
                va="top", ha="center")


def plot_panel(ax, data, title):
    for name, entry in data.items():
        color, sp, ens = entry[0], entry[1], entry[2]
        sp_pos, ens_pos = entry[3], entry[4]
        sp_x, sp_y = sp
        ax.scatter(sp_x, sp_y, color=color, s=70, marker="o",
                   edgecolor="black", linewidth=0.5, zorder=3)
        if ens is not None:
            ens_x, ens_y = ens
            ax.scatter(ens_x, ens_y, color=color, s=110, marker="D",
                       edgecolor="black", linewidth=0.7, zorder=3)
            ax.annotate("", xy=(ens_x, ens_y), xytext=(sp_x, sp_y),
                        arrowprops=dict(arrowstyle="->", color=color,
                                        alpha=0.55, lw=1.5),
                        zorder=2)
            _place(ax, sp_x, sp_y + sp_pos[1], name, color, sp_pos[0])
            _place(ax, ens_x, ens_y + ens_pos[1],
                   f"{name} +Self-ens.", color, ens_pos[0])
        else:
            _place(ax, sp_x, sp_y + sp_pos[1], name, color, sp_pos[0])

    ax.set_xscale("log")
    ax.set_xlabel("Cost per image (millicents, USD)")
    ax.set_ylabel(r"RMS$_{F1}$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(FIG_W, FIG_H),
                                    constrained_layout=True)
    plot_panel(axL, CQ, "ChartQA")
    plot_panel(axR, WB, "WB-ChartExtract")

    # ChartQA y-range covers ~75-95
    axL.set_ylim(72, 96)
    axL.set_xlim(8, 8000)
    # WB y-range covers ~28-90
    axR.set_ylim(26, 92)
    axR.set_xlim(20, 8000)

    # Free-model annotations (TinyChart/OneChart can't sit on a log-cost axis)
    free_box = dict(boxstyle="round,pad=0.3", facecolor="white",
                    edgecolor="gray", linewidth=0.5, alpha=0.9)
    axL.text(0.98, 0.02,
             "Free (not shown):\nTinyChart 95.20\nDePlot 88.32\nOneChart 35.93",
             transform=axL.transAxes, fontsize=7, va="bottom", ha="right",
             bbox=free_box)
    axR.text(0.02, 0.55,
             "Free (not shown):\nTinyChart 28.61\nOneChart 26.49\nDePlot 23.06",
             transform=axR.transAxes, fontsize=7, va="bottom", ha="left",
             bbox=free_box)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="black", markersize=9, label="Single-pass"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="gray",
               markeredgecolor="black", markersize=10, label="Self-ensembled"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               frameon=True, fancybox=False, framealpha=0.95,
               bbox_to_anchor=(0.5, -0.03), bbox_transform=fig.transFigure)
    fig.get_layout_engine().set(rect=(0, 0.05, 1, 1))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=600, facecolor="white", bbox_inches="tight")
    pdf = OUTPUT.with_suffix(".pdf")
    fig.savefig(pdf, dpi=600, facecolor="white", bbox_inches="tight")
    print(f"Saved {OUTPUT}")
    print(f"Saved {pdf}")


if __name__ == "__main__":
    main()
