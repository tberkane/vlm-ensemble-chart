"""Regenerate convergence_rate_comparison.pdf for WB v2 (and ChartQA when available)."""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

ACL_COL_WIDTH_IN = 3.25
ACL_ASPECT = 0.55
FIG_W = ACL_COL_WIDTH_IN
FIG_H = ACL_COL_WIDTH_IN * ACL_ASPECT * 2

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Roboto", "DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
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
})

WB_RUN = "outputs/adaptive/wb_v2_scout_em1_prune0p2"
CHARTQA_RUN = "outputs/adaptive/2026-04-21_095020__ChartQA_Dataset_test__meta-llama_llama-4-scout-17b-16e-instruct_temp2p0_maxsamp20_pat2_median"

OUTPUT = Path("outputs/experiments/convergence_rate_comparison.pdf")


def load(run_dir):
    p = Path(run_dir) / "iteration_metrics.json"
    if not p.exists():
        return None, None, None
    im = json.load(open(p))
    ks = sorted(int(k) for k in im.keys())
    f1s = [im[str(k)]["f1"] for k in ks]
    convs = [im[str(k)].get("convergence_rate", 0) for k in ks]
    return ks, f1s, convs


def find_median_stopping_point(ks, convs):
    for k, c in zip(ks, convs):
        if c >= 50.0:
            return k
    return None


def plot_panel(ax, ks, f1s, convs, title):
    f1_color = "#648fff"
    conv_color = "#dc267f"

    ax.plot(ks, f1s, color=f1_color, marker="o", linestyle="-",
            linewidth=1.2, markersize=4.0, markeredgewidth=0.4, markeredgecolor="white")
    ax.set_ylabel(r"RMS$_{F_1}$", color=f1_color)
    ax.tick_params(axis="y", labelcolor=f1_color)

    ax_conv = ax.twinx()
    ax_conv.plot(ks, convs, color=conv_color, marker="s", linestyle="--",
                 linewidth=1.2, markersize=4.0, markeredgewidth=0.4, markeredgecolor="white")
    ax_conv.set_ylabel("Conv. %", color=conv_color)
    ax_conv.tick_params(axis="y", labelcolor=conv_color)
    ax_conv.set_ylim(0, 100)

    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xticks([3, 5, 7, 10, 15, 20])
    ax.set_xlim(2.5, 20.5)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)

    med = find_median_stopping_point(ks, convs)
    if med is not None:
        ax.axvline(x=med, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)


def plot():
    plt.style.use("default")

    cqa = load(CHARTQA_RUN)
    wb = load(WB_RUN)

    have_cqa = cqa[0] is not None
    if have_cqa:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_W, FIG_H), constrained_layout=True, sharex=True)
        plot_panel(ax1, *cqa, "ChartQA")
        plot_panel(ax2, *wb, "WB-ChartExtract")
        ax2.set_xlabel("Ensemble Size $K$")
    else:
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H / 2), constrained_layout=True)
        plot_panel(ax, *wb, "WB-ChartExtract")
        ax.set_xlabel("Ensemble Size $K$")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=600, facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=600, facecolor="white")
    print(f"Saved {OUTPUT}  (ChartQA panel: {'yes' if have_cqa else 'no'})")


if __name__ == "__main__":
    plot()
