#!/usr/bin/env python3
"""
Create scatter plot of mean MAD (median absolute deviation) vs F1 score.

For each ensembled table, computes the mean relative MAD across all cells
and the F1 score for the table, then plots mean MAD vs F1 score.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from jsonargparse import CLI
from scipy.stats import spearmanr

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.eval_chart2table import Table, _parse_table, _to_float, chart2table_evaluator
from src.utils import normalize_tsv_table

# Configure matplotlib for publication-quality plots
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
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 11,
        "text.usetex": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load_predictions(predictions_path: Path) -> List[Dict]:
    """Load predictions from JSON file."""
    with predictions_path.open("r", encoding="utf-8") as f:
        predictions = json.load(f)
    return predictions


def load_ground_truth_tables(gt_dir: Path) -> Dict[str, str]:
    """Load ground truth tables from CSV files and convert to TSV."""
    import csv
    import io

    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")

    tables_dict = {}
    for csv_file in gt_dir.glob("*.csv"):
        with csv_file.open("r", encoding="utf-8") as f:
            csv_content = f.read()
            # Convert CSV to TSV
            reader = csv.reader(io.StringIO(csv_content))
            output = io.StringIO()
            writer = csv.writer(output, delimiter="\t")
            for row in reader:
                writer.writerow(row)
            tsv_content = output.getvalue()
            tsv_content = normalize_tsv_table(tsv_content)
            tables_dict[csv_file.stem] = tsv_content

    return tables_dict


def parse_table_to_cell_dict(tsv: str) -> Dict[Tuple[str, str], Tuple[int, int, str]]:
    """
    Parse TSV table and return dict mapping (row_label, col_label) -> (row_idx, col_idx, value).

    Returns:
        Dict mapping (row_label, col_label) to (row_idx, col_idx, cell_value)
    """
    if not tsv or not tsv.strip():
        return {}

    table = _parse_table(tsv, transposed=False)
    cell_dict = {}

    if not table.headers or not table.rows:
        return cell_dict

    # Build mapping: (row_label, col_label) -> (row_idx, col_idx, value)
    for row_idx, row in enumerate(table.rows):
        if not row:
            continue
        row_label = row[0].strip()

        # Match columns with headers (skip first header which is usually empty or row label)
        for col_idx, header in enumerate(table.headers[1:], start=1):
            if col_idx < len(row):
                col_label = header.strip()
                cell_value = row[col_idx].strip()
                cell_dict[(row_label, col_label)] = (row_idx, col_idx, cell_value)

    return cell_dict


def parse_mad_tsv_to_position_map(tsv: str) -> Dict[Tuple[int, int], float]:
    """
    Parse MAD TSV table and return dict mapping (row_idx, col_idx) -> mad_value.

    Returns:
        Dict mapping (row_idx, col_idx) to MAD value as float
    """
    if not tsv or not tsv.strip():
        return {}

    table = _parse_table(tsv, transposed=False)
    mad_map = {}

    if not table.headers or not table.rows:
        return mad_map

    # Build mapping: (row_idx, col_idx) -> mad_value
    for row_idx, row in enumerate(table.rows):
        if not row:
            continue

        # Skip first column (row label), start from col_idx=1
        for col_idx in range(1, len(row)):
            mad_str = row[col_idx].strip()
            mad_val = _to_float(mad_str)
            if mad_val is not None:
                mad_map[(row_idx, col_idx)] = mad_val
            else:
                # If parsing fails, default to 0.0
                mad_map[(row_idx, col_idx)] = 0.0

    return mad_map


def compute_relative_error(pred_value: float, gt_value: float, eps: float) -> float:
    """Relative error: |pred - gt| / (|gt| + eps)."""
    return abs(pred_value - gt_value) / (abs(gt_value) + eps)


def compute_relative_mad(
    mad_value: float, pred_scale_value: float, eps: float
) -> float:
    """Relative MAD: MAD / (|pred_scale| + eps). Uses ensembled pred as scale proxy."""
    return mad_value / (abs(pred_scale_value) + eps)


def compute_per_image_f1(pred_tsv: str, gt_tsv: str) -> float:
    """Compute F1 score for a single prediction-ground truth pair."""
    result = chart2table_evaluator(
        [
            {
                "model_answer": pred_tsv,
                "gt_answer": gt_tsv,
                "prediction_index": 0,
                "image": "temp",
            }
        ],
        disable_tqdm=True,
    )
    return result


def extract_per_image_statistics(
    predictions: List[Dict],
    ground_truth: Dict[str, str],
    *,
    eps: float = 1e-6,
) -> Dict[str, Dict[str, Any]]:
    """
    Extract per-image statistics for MAD and relative error.

    Returns:
        Dictionary mapping image_name -> {
            'mean_mad': mean relative MAD across all cells,
            'mean_rel_error': mean relative error across all cells,
            'f1': F1 score for the table,
            'mean_mad_cell': dict with pred_value, gt_value, mad_value, pred_scale_value for mean MAD cell,
            'max_error_cell': dict with pred_value, gt_value for max error cell,
        }
    """
    from collections import defaultdict

    image_stats: Dict[str, Dict[str, List]] = defaultdict(
        lambda: {"mads": [], "errors": [], "mad_cells": [], "error_cells": []}
    )
    # Store pred_tsv and gt_tsv for each image for F1 computation
    image_tables: Dict[str, Dict[str, str]] = {}

    for pred in predictions:
        image_name = pred.get("image", "").replace(".png", "")
        if image_name not in ground_truth:
            continue

        pred_tsv = pred.get("answer", "")
        if not pred_tsv:
            continue

        gt_tsv = ground_truth[image_name]
        mad_tsv = pred.get("mad_tsv", "")
        if not mad_tsv:
            continue

        # Store tables for F1 computation
        image_tables[image_name] = {"pred_tsv": pred_tsv, "gt_tsv": gt_tsv}

        mad_map = parse_mad_tsv_to_position_map(mad_tsv)
        pred_cells = parse_table_to_cell_dict(pred_tsv)
        gt_cells = parse_table_to_cell_dict(gt_tsv)

        for (row_label, col_label), (
            row_idx,
            col_idx,
            pred_val_str,
        ) in pred_cells.items():
            if (row_label, col_label) not in gt_cells:
                continue

            _, _, gt_val_str = gt_cells[(row_label, col_label)]
            mad = mad_map.get((row_idx, col_idx), 0.0)

            pred_val = _to_float(pred_val_str)
            gt_val = _to_float(gt_val_str)

            if pred_val is None or gt_val is None:
                continue

            # Filter out points where gt_value or pred_scale_value is 0
            if gt_val == 0 or pred_val == 0:
                continue

            # Compute relative error and relative MAD
            rel_err = compute_relative_error(pred_val, gt_val, eps=eps)
            rel_mad = compute_relative_mad(mad, pred_scale_value=pred_val, eps=eps)

            image_stats[image_name]["mads"].append(rel_mad)
            image_stats[image_name]["errors"].append(rel_err)
            # Store cell values for max MAD tracking
            image_stats[image_name]["mad_cells"].append(
                {
                    "rel_mad": rel_mad,
                    "pred_value": pred_val,
                    "gt_value": gt_val,
                    "mad_value": mad,
                    "pred_scale_value": pred_val,
                }
            )
            # Store cell values for max error tracking
            image_stats[image_name]["error_cells"].append(
                {
                    "rel_error": rel_err,
                    "pred_value": pred_val,
                    "gt_value": gt_val,
                }
            )

    # Aggregate statistics per image
    result: Dict[str, Dict[str, Any]] = {}
    for image_name, stats in image_stats.items():
        if stats["mads"] and stats["errors"]:
            # Find cell with mean MAD
            mean_mad_value = np.mean(stats["mads"])
            # Find the cell closest to the mean MAD value
            mean_mad_cell = min(
                stats["mad_cells"], key=lambda x: abs(x["rel_mad"] - mean_mad_value)
            )
            # Find cell with max error
            max_error_cell = max(stats["error_cells"], key=lambda x: x["rel_error"])

            # Compute F1 score for this image
            f1_score = 0.0
            if image_name in image_tables:
                pred_tsv = image_tables[image_name]["pred_tsv"]
                gt_tsv = image_tables[image_name]["gt_tsv"]
                try:
                    f1_score = compute_per_image_f1(pred_tsv, gt_tsv)
                except Exception as e:
                    print(f"Warning: Failed to compute F1 for {image_name}: {e}")
                    f1_score = 0.0

            result[image_name] = {
                "mean_mad": mean_mad_value,
                "median_mad": np.median(stats["mads"]),
                "max_mad": np.max(stats["mads"]),
                "mean_rel_error": np.mean(stats["errors"]),
                "f1": f1_score,
                "mean_mad_cell": {
                    "pred_value": mean_mad_cell["pred_value"],
                    "gt_value": mean_mad_cell["gt_value"],
                    "mad_value": mean_mad_cell["mad_value"],
                    "pred_scale_value": mean_mad_cell["pred_scale_value"],
                },
                "max_error_cell": {
                    "pred_value": max_error_cell["pred_value"],
                    "gt_value": max_error_cell["gt_value"],
                },
            }

    return result


def create_mean_mad_vs_f1_plot(
    mean_mad_values: List[float],
    f1_scores: List[float],
    output_path: Path,
    title: str = "Mean MAD vs F1 Score",
):
    """Create scatter plot of MAD statistic vs F1 score per table."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        mean_mad_values,
        f1_scores,
        alpha=0.5,
        s=20,
        edgecolors="none",
    )

    # Determine label based on title
    if "Median" in title:
        xlabel = "Median Relative MAD = median(MAD / (|pred| + eps)) per table"
    elif "Max" in title:
        xlabel = "Max Relative MAD = max(MAD / (|pred| + eps)) per table"
    else:
        xlabel = "Mean Relative MAD = mean(MAD / (|pred| + eps)) per table"

    ax.set_xlabel(xlabel)
    ax.set_ylabel("F1 Score")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # log scaling for MAD if needed
    if mean_mad_values and max(mean_mad_values) > 0:
        pos = [m for m in mean_mad_values if m > 0]
        if pos:
            mn = min(pos)
            mx = max(mean_mad_values)
            if mx / mn > 100:
                ax.set_xscale("log")

    if mean_mad_values and f1_scores:
        corr, p_value = spearmanr(mean_mad_values, f1_scores)
        stats_text = f"n = {len(mean_mad_values)}\nSpearman ρ = {corr:.3f}"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close()


def main(
    predictions_path: str,
    gt_dir: str,
    output_path: str = "outputs/experiments/mad_vs_error.png",
    eps: float = 1e-6,
):
    """
    Create scatter plot of mean MAD vs F1 score per table.

    Args:
        predictions_path: Path to predictions.json file
        gt_dir: Path to ground truth directory (CSV files)
        output_path: Path to save the plot (used as base path for mean_mad_vs_f1 plot)
        eps: Small constant for normalization stability
    """
    predictions_path = Path(predictions_path)
    gt_dir = Path(gt_dir)
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading predictions from {predictions_path}...")
    predictions = load_predictions(predictions_path)
    print(f"Loaded {len(predictions)} predictions")

    print(f"Loading ground truth from {gt_dir}...")
    ground_truth = load_ground_truth_tables(gt_dir)
    print(f"Loaded {len(ground_truth)} ground truth tables")

    # Extract per-image statistics and print top 5 for each metric
    print("\nExtracting per-image statistics...")
    image_stats = extract_per_image_statistics(predictions, ground_truth, eps=eps)

    # Create plots for median, mean, and max MAD vs F1
    print("\nComputing MAD statistics vs F1 score...")
    median_mad_values = []
    mean_mad_values = []
    max_mad_values = []
    f1_scores = []
    for image_name, stats in image_stats.items():
        if (
            "mean_mad" in stats
            and "median_mad" in stats
            and "max_mad" in stats
            and "f1" in stats
        ):
            median_mad_values.append(stats["median_mad"])
            mean_mad_values.append(stats["mean_mad"])
            max_mad_values.append(stats["max_mad"])
            f1_scores.append(stats["f1"])

    if mean_mad_values and f1_scores:
        # Create plots for each statistic
        for mad_type, mad_values, mad_label in [
            ("median", median_mad_values, "Median"),
            ("mean", mean_mad_values, "Mean"),
            ("max", max_mad_values, "Max"),
        ]:
            # Create output path for the plot
            mad_f1_output_path = output_path.parent / (
                output_path.stem + f"_{mad_type}_mad_vs_f1" + output_path.suffix
            )
            print(f"Creating {mad_label} MAD vs F1 plot...")
            create_mean_mad_vs_f1_plot(
                mad_values,
                f1_scores,
                mad_f1_output_path,
                f"{mad_label} MAD vs F1 Score",
            )

        # Compute and print Spearman correlations
        print(f"\nMAD Statistics vs F1 Score (Spearman Correlation):")
        print(f"  Number of tables: {len(mean_mad_values)}")

        for mad_type, mad_values, mad_label in [
            ("median", median_mad_values, "Median"),
            ("mean", mean_mad_values, "Mean"),
            ("max", max_mad_values, "Max"),
        ]:
            corr, p_value = spearmanr(mad_values, f1_scores)
            print(f"\n  {mad_label} MAD vs F1:")
            print(
                f"    {mad_label} MAD - mean: {np.mean(mad_values):.6f}, median: {np.median(mad_values):.6f}, max: {np.max(mad_values):.6f}"
            )
            print(f"    Spearman correlation (ρ): {corr:.4f} (p-value: {p_value:.4f})")

        print(
            f"\n  F1 Score - mean: {np.mean(f1_scores):.6f}, median: {np.median(f1_scores):.6f}, max: {np.max(f1_scores):.6f}"
        )
    else:
        print("Warning: No MAD statistics or F1 scores found for plotting.")


if __name__ == "__main__":
    CLI(main)
