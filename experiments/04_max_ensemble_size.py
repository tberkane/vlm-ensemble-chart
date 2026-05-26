import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

# =========================
# ACL sizing (single column)
# =========================
ACL_COL_WIDTH_IN = 3.25  # typical \columnwidth in ACL templates
ACL_ASPECT = 0.60  # height = width * aspect; tweak 0.60–0.85 as needed
FIG_W = ACL_COL_WIDTH_IN
FIG_H = ACL_COL_WIDTH_IN * ACL_ASPECT

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


def load_data(output_dir: Path):
    """Load iteration metrics and statistics from output directory."""
    iteration_metrics_path = output_dir / "iteration_metrics.json"
    statistics_path = output_dir / "statistics.json"

    if not iteration_metrics_path.exists():
        raise FileNotFoundError(f"iteration_metrics.json not found in {output_dir}")

    if not statistics_path.exists():
        raise FileNotFoundError(f"statistics.json not found in {output_dir}")

    with iteration_metrics_path.open() as f:
        iteration_metrics = json.load(f)

    with statistics_path.open() as f:
        statistics = json.load(f)

    total_images = statistics["total_images"]

    # Extract data sorted by ensemble size
    ensemble_sizes = []
    f1_scores = []
    convergence_rates = []

    for size_str, metrics in sorted(iteration_metrics.items(), key=lambda x: int(x[0])):
        size = int(size_str)
        f1 = metrics["f1"]
        convergence_rate = metrics["convergence_rate"]

        ensemble_sizes.append(size)
        f1_scores.append(f1)
        convergence_rates.append(convergence_rate)

    return ensemble_sizes, f1_scores, convergence_rates


def plot_max_ensemble_size(output_dir: Path, output_plot: Path = None):
    """Create plot with F1 and convergence rate vs max ensemble size."""
    plt.style.use("default")

    ensemble_sizes, f1_scores, convergence_rates = load_data(output_dir)

    # Use constrained_layout for stable sizing (avoid bbox_inches="tight" shrink-wrapping)
    fig, ax1 = plt.subplots(figsize=(FIG_W, FIG_H), constrained_layout=True)

    # Plot F1 on left y-axis
    color_f1 = "#648fff"
    ax1.set_xlabel("Max Ensemble Size")
    ax1.set_ylabel("F1", color=color_f1)
    line1 = ax1.plot(
        ensemble_sizes,
        f1_scores,
        color=color_f1,
        linestyle="-",
        linewidth=1.2,
        label="F1",
    )
    ax1.tick_params(axis="y", labelcolor=color_f1)
    ax1.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax1.set_axisbelow(True)

    # Create second y-axis for convergence rate
    ax2 = ax1.twinx()
    color_conv = "#ffb000"
    ax2.set_ylabel("Convergence Rate (%)", color=color_conv)
    line2 = ax2.plot(
        ensemble_sizes,
        convergence_rates,
        color=color_conv,
        linestyle="--",
        linewidth=1.2,
        label="Convergence Rate",
    )
    ax2.tick_params(axis="y", labelcolor=color_conv)

    # Set x-axis limits and ticks (avoid overcrowded x-axis labels)
    if ensemble_sizes:
        min_size = min(ensemble_sizes)
        max_size = max(ensemble_sizes)
        ax1.set_xlim(min_size - 0.5, max_size + 0.5)

        # Determine spacing to avoid overcrowding
        max_labels = 12  # maximum number of x-tick labels for clarity
        num_ensembles = max_size - min_size + 1
        if num_ensembles <= max_labels:
            tick_step = 1
        else:
            # Choose an integer step to reduce overcrowding
            tick_step = int(round(num_ensembles / max_labels))
            # Ensure at least 1
            tick_step = max(1, tick_step)

        xticks = list(range(min_size, max_size + 1, tick_step))
        # Always ensure last value is present as xtick
        if xticks[-1] != max_size:
            xticks.append(max_size)
        ax1.set_xticks(xticks)

    # Style the axes
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax1.spines["left"].set_linewidth(0.6)
    ax1.spines["bottom"].set_linewidth(0.6)
    ax2.spines["right"].set_linewidth(0.6)
    ax2.spines["bottom"].set_visible(False)

    # Create combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    # ax1.legend(
    #     lines,
    #     labels,
    #     loc="best",
    #     frameon=True,
    #     fancybox=False,
    #     framealpha=0.95,
    #     borderpad=0.3,
    #     handlelength=2.0,
    #     handletextpad=0.6,
    #     labelspacing=0.3,
    # )

    # Save plot
    if output_plot is None:
        output_plot = output_dir / "max_ensemble_size_plot.png"

    output_plot.parent.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: omit bbox_inches="tight" to preserve the intended physical size
    fig.savefig(output_plot, dpi=600, facecolor="white")
    pdf_path = output_plot.with_suffix(".pdf")
    fig.savefig(pdf_path, dpi=600, facecolor="white")
    print(f"Saved plot to {output_plot}")
    print(f"Saved plot to {pdf_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 04_max_ensemble_size.py <output_dir> [output_plot]")
        print(
            "Example: python 04_max_ensemble_size.py outputs/adaptive/2025-12-28_042458__png__incremental_ensemble_meta-llama_llama-4-maverick-17b-128e-instruct"
        )
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    if not output_dir.exists():
        print(f"Error: Directory {output_dir} does not exist")
        sys.exit(1)

    output_plot = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    plot_max_ensemble_size(output_dir, output_plot)
