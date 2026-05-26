"""Regenerate temperature_ablation.pdf using WB v2 + Scout data.

Plots WB-ChartExtract v2 F1 vs ensemble size K for T=0.0, 1.0, 2.0.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

ACL_COL_WIDTH_IN = 3.25
ACL_ASPECT = 0.70
FIG_W = ACL_COL_WIDTH_IN
FIG_H = ACL_COL_WIDTH_IN * ACL_ASPECT * 2  # two panels stacked

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

WB_T_DIRS = {
    "T=0.0": "outputs/adaptive/wb_v2_scout_em1_prune0p2",
    "T=1.0": "outputs/adaptive/2026-04-13_130605__World_Bank_v2_test__meta-llama_llama-4-scout-17b-16e-instruct_temp1p0_maxsamp20_pat2_medoid",
    "T=2.0": "outputs/adaptive/2026-04-13_130605__World_Bank_v2_test__meta-llama_llama-4-scout-17b-16e-instruct_temp2p0_maxsamp20_pat2_medoid",
}

import glob
CQA_T0 = sorted(glob.glob("outputs/adaptive/*ChartQA*temp0p0*median"))[-1]
CQA_T1 = sorted(glob.glob("outputs/adaptive/*ChartQA*temp1p0*median"))[-1]
CQA_T2 = sorted(glob.glob("outputs/adaptive/*ChartQA*temp2p0*median"))[-1]
CQA_T_DIRS = {"T=0.0": CQA_T0, "T=1.0": CQA_T1, "T=2.0": CQA_T2}

T_DIRS = WB_T_DIRS  # default (back-compat)

OUTPUT = Path("outputs/experiments/temperature_ablation.pdf")


def load_f1_by_k(run_dir: str):
    im = json.load(open(Path(run_dir) / "iteration_metrics.json"))
    ks = sorted(int(k) for k in im.keys())
    f1s = [im[str(k)]["f1"] for k in ks]
    return ks, f1s


def _plot_panel(ax, dirs, title):
    colors = ["#648fff", "#ffb000", "#dc267f"]
    linestyles = ["-", "--", "-."]
    markers = ["o", "s", "^"]
    for (label, run_dir), color, ls, m in zip(dirs.items(), colors, linestyles, markers):
        ks, f1s = load_f1_by_k(run_dir)
        ax.plot(ks, f1s, label=r"$" + label[1:] + r"$", color=color, marker=m,
                linestyle=ls, linewidth=1.2, markersize=4.0,
                markeredgewidth=0.4, markeredgecolor="white")
    ax.set_ylabel(r"RMS$_{F_1}$")
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xticks([3, 5, 7, 10, 15, 20])
    ax.set_xlim(2.5, 20.5)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=True, fancybox=False, framealpha=0.95,
              borderpad=0.3, handlelength=2.0, handletextpad=0.6, labelspacing=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)


def plot():
    plt.style.use("default")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_W, FIG_H), constrained_layout=True, sharex=True)
    _plot_panel(ax1, CQA_T_DIRS, "ChartQA")
    _plot_panel(ax2, WB_T_DIRS, "WB-ChartExtract")
    ax2.set_xlabel("Ensemble Size $K$")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=600, facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=600, facecolor="white")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    plot()
