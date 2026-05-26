#!/usr/bin/env python3
"""
Ensemble Size vs F1 Score

Measure how ensemble size affects F1 score when aggregating predictions
from N model runs. For each ensemble size from 1 to N, we:
1. Sample subsets of size `num_members` (with or without replacement)
2. Build ensemble predictions using median aggregation
3. Compute F1 scores
4. Calculate mean F1 and bootstrap 95% confidence intervals
"""

# import cProfile
import csv
import io
import json
import logging
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from jsonargparse import CLI
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from scripts.ensemble import aggregate_answers, tsvs_to_dfs
from src.eval_chart2table import chart2table_evaluator
from src.utils import normalize_tsv_table

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s %(asctime)s %(module)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class EnsembleSizeExperimentConfig:

    # Directories of base prediction runs (N runs)
    member_run_dirs: List[str]

    # Ground truth directory
    gt_dir: str

    # Number of repetitions per ensemble size
    num_repetitions: int = 10_000

    sample_with_replacement: bool = False

    aggregation_methods: List[str] = field(default_factory=lambda: ["median"])

    # Output directory for results
    output_dir: Optional[str] = None


def load_predictions_from_runs(
    member_dirs: List[Path],
) -> Dict[str, List[Optional[str]]]:
    """
    Load predictions from multiple run directories.

    Returns:
        Dictionary mapping image name (without extension) -> list of predictions
        Structure: {image_name: [answer_from_run0, answer_from_run1, ..., answer_from_runN-1]}
    """
    num_runs = len(member_dirs)
    predictions_by_image: Dict[str, List[Optional[str]]] = defaultdict(
        lambda: [None] * num_runs
    )

    for run_idx, run_dir in enumerate(member_dirs):
        pred_path = run_dir / "predictions.json"
        if not pred_path.exists():
            raise FileNotFoundError(f"predictions.json not found in {run_dir}")

        with pred_path.open("r", encoding="utf-8") as f:
            preds = json.load(f)

        for entry in preds:
            img = entry["image"]
            img_key = img.split(".")[0]
            ans = entry["answer"]
            predictions_by_image[img_key][run_idx] = ans

    # Filter to only keep images with predictions from all runs
    complete_predictions_by_image = {
        img: preds
        for img, preds in predictions_by_image.items()
        if all(p is not None for p in preds)
    }

    logger.info(
        f"Loaded predictions from {num_runs} runs for {len(predictions_by_image)} images"
    )
    logger.info(
        f"  {len(complete_predictions_by_image)} images have predictions from all {num_runs} runs"
    )

    return complete_predictions_by_image


def load_ground_truth_tables(gt_dir: Path) -> Dict[str, str]:
    """Load ground truth tables from CSV files."""
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")

    tables_dict = {}
    for csv_file in gt_dir.glob("*.csv"):
        with csv_file.open("r", encoding="utf-8") as f:
            csv_content = f.read()
            # Convert CSV to TSV
            reader = csv.reader(io.StringIO(csv_content))
            output = io.StringIO()
            writer = csv.writer(output, delimiter="\t", lineterminator="\n")
            for row in reader:
                writer.writerow(row)
            tsv_content = output.getvalue()
            tsv_content = normalize_tsv_table(tsv_content)
            tables_dict[csv_file.name.split(".")[0]] = tsv_content

    logger.info(f"Loaded {len(tables_dict)} ground truth tables from {gt_dir}")
    return tables_dict


def compute_f1_for_ensemble(
    ensemble_predictions: Dict[str, str],
    ground_truth_tables: Dict[str, str],
) -> float:
    """
    Compute F1 score for ensemble predictions.

    Args:
        ensemble_predictions: Dictionary mapping image name -> ensemble prediction (TSV)
        ground_truth_tables: Dictionary mapping image name -> ground truth table (TSV)

    Returns:
        F1 score (float)
    """
    preds_and_gts = []
    for image, prediction in ensemble_predictions.items():
        table = ground_truth_tables.get(image)
        if table is None:
            continue

        # Replace multiple contiguous spaces with \t
        prediction_processed = re.sub(r"[ ]{2,}", "\t", prediction)

        preds_and_gts.append(
            {
                "model_answer": prediction_processed,
                "gt_answer": table,
                "image": image,
            }
        )

    if not preds_and_gts:
        logger.warning("No matching predictions and ground truth found")
        return 0.0

    f1 = chart2table_evaluator(preds_and_gts, disable_tqdm=True)
    return f1


def bootstrap_confidence_interval(
    data: np.ndarray, confidence: float = 0.95, n_samples: int = 10_000
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for the mean.

    Args:
        data: Array of F1 scores
        confidence: Confidence level (default 0.95 for 95% CI)
        n_samples: Number of bootstrap samples

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(data) == 0:
        return (0.0, 0.0)

    means = []
    for _ in range(n_samples):
        # Sample with replacement
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))

    means = np.array(means)
    alpha = 1 - confidence
    lower = np.percentile(means, 100 * alpha / 2)
    upper = np.percentile(means, 100 * (1 - alpha / 2))

    return (lower, upper)


def sample_run_indices(
    num_runs: int, num_members: int, with_replacement: bool
) -> List[int]:
    """
    Sample indices for ensemble members.

    Args:
        num_runs: Total number of runs available
        num_members: Number of members to sample
        with_replacement: Whether to sample with replacement

    Returns:
        List of run indices
    """
    if with_replacement:
        return list(np.random.choice(num_runs, size=num_members, replace=True))
    else:
        if num_members > num_runs:
            raise ValueError(
                f"Cannot sample {num_members} members without replacement from {num_runs} runs"
            )
        return list(np.random.choice(num_runs, size=num_members, replace=False))


def main(cfg: EnsembleSizeExperimentConfig) -> None:
    """Main experiment function."""
    # profiler = cProfile.Profile()
    # profiler.enable()

    # 1. Normalize paths
    project_root = Path(__file__).resolve().parents[1]

    member_dirs: List[Path] = []
    for d in cfg.member_run_dirs:
        p = Path(d)
        if not p.is_absolute():
            p = project_root / p
        if not p.exists():
            raise FileNotFoundError(f"Member run dir does not exist: {p}")
        member_dirs.append(p)

    gt_dir = Path(cfg.gt_dir)
    if not gt_dir.is_absolute():
        gt_dir = project_root / gt_dir

    num_runs = len(member_dirs)
    logger.info(f"Experiment configuration:")
    logger.info(f"  Number of base runs (N): {num_runs}")
    logger.info(f"  Repetitions per ensemble size: {cfg.num_repetitions}")
    logger.info(f"  Sample with replacement: {cfg.sample_with_replacement}")
    logger.info(f"  Aggregation methods: {cfg.aggregation_methods}")

    # 2. Load predictions and ground truth
    logger.info("Loading predictions and ground truth...")
    predictions_by_image = load_predictions_from_runs(member_dirs)
    predictions_by_image_dfs = {
        k: tsvs_to_dfs(v) for k, v in predictions_by_image.items()
    }
    ground_truth_tables = load_ground_truth_tables(gt_dir)

    # Filter to images that have both predictions and ground truth
    common_images = set(predictions_by_image_dfs.keys()) & set(
        ground_truth_tables.keys()
    )
    logger.info(
        f"Found {len(common_images)} images with both predictions and ground truth"
    )

    # 3. Prepare output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    if cfg.output_dir:
        output_dir = Path(cfg.output_dir)
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir
        # Prepend timestamp to the directory name
        output_dir = output_dir.parent / f"{timestamp}__{output_dir.name}"
    else:
        output_dir = (
            project_root / "outputs" / "experiments" / f"{timestamp}__ensemble_size"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # 4. Run experiment for each ensemble size
    # Structure: results[method][num_members] = {...}
    results = {method: {} for method in cfg.aggregation_methods}
    all_f1_scores = {
        method: {} for method in cfg.aggregation_methods
    }  # method -> num_members -> list of F1 scores

    for num_members in range(1, num_runs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Ensemble size: {num_members}")
        logger.info(f"{'='*60}")

        # Initialize F1 scores for each method
        f1_scores_by_method = {method: [] for method in cfg.aggregation_methods}

        np.random.seed(42 + num_members)
        random.seed(42 + num_members)

        for rep in tqdm(
            range(cfg.num_repetitions), desc=f"Repetitions (size={num_members})"
        ):
            try:
                run_indices = sample_run_indices(
                    num_runs, num_members, cfg.sample_with_replacement
                )
            except ValueError as e:
                logger.error(f"Error sampling: {e}")
                continue

            # Initialize ensemble_predictions as nested dict: method -> image -> prediction
            ensemble_predictions_by_method = {
                method: {} for method in cfg.aggregation_methods
            }

            for image in common_images:
                # Get predictions from selected runs
                # predictions_by_image[image] is a list: [answer_from_run0, answer_from_run1, ...]
                selected_predictions = []
                for run_idx in run_indices:
                    if run_idx < len(predictions_by_image_dfs[image]):
                        pred = predictions_by_image_dfs[image][run_idx]
                        if pred is not None:
                            selected_predictions.append(pred)

                if not selected_predictions:
                    continue
                ensembled_answers = aggregate_answers(
                    selected_predictions,
                    cfg.aggregation_methods,
                )
                for method in cfg.aggregation_methods:
                    if method in ensembled_answers:
                        ensemble_predictions_by_method[method][image] = (
                            ensembled_answers[method]
                        )

            # Compute F1 scores for each method separately
            for method in cfg.aggregation_methods:
                if ensemble_predictions_by_method[method]:
                    f1 = compute_f1_for_ensemble(
                        ensemble_predictions_by_method[method], ground_truth_tables
                    )
                    f1_scores_by_method[method].append(f1)

        # Process results for each method
        for method in cfg.aggregation_methods:
            f1_scores = f1_scores_by_method[method]

            if not f1_scores:
                logger.warning(
                    f"No F1 scores computed for ensemble size {num_members}, method {method}"
                )
                continue

            mean_f1 = np.mean(f1_scores)
            std_f1 = np.std(f1_scores)
            ci_lower, ci_upper = bootstrap_confidence_interval(
                np.array(f1_scores), n_samples=cfg.num_repetitions
            )

            results[method][num_members] = {
                "mean_f1": float(mean_f1),
                "std_f1": float(std_f1),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "num_repetitions": len(f1_scores),
            }
            all_f1_scores[method][num_members] = f1_scores

            logger.info(f"  Method {method}:")
            logger.info(f"    Mean F1: {mean_f1:.4f}")
            logger.info(f"    Std F1: {std_f1:.4f}")
            logger.info(f"    95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    # profiler.disable()
    # profile_file = (
    #     project_root
    #     / "outputs"
    #     / "profiles"
    #     / f"ensemble_size_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prof"
    # )
    # profile_file.parent.mkdir(parents=True, exist_ok=True)
    # profiler.dump_stats(str(profile_file))

    # 5. Create plot
    logger.info("\nCreating plot...")
    plt.figure(figsize=(10, 6))

    # Plot one line per aggregation method
    # Use colorblind-friendly palette (Color Universal Design - CUD)
    colorblind_palette = [
        "#0072B2",  # blue
        "#D55E00",  # vermillion
        "#F0E442",  # yellow
        "#009E73",  # green
        "#CC79A7",  # purple
        "#56B4E9",  # light blue
        "#E69F00",  # orange
        "#000000",  # black
        "#999999",  # gray
        "#DC3220",  # red (Solarized)
    ]
    # Cycle if not enough colors
    colors = [
        colorblind_palette[i % len(colorblind_palette)]
        for i in range(len(cfg.aggregation_methods))
    ]

    for method_idx, method in enumerate(cfg.aggregation_methods):
        if not results[method]:
            continue

        ensemble_sizes = sorted(results[method].keys())
        mean_f1s = [results[method][s]["mean_f1"] for s in ensemble_sizes]
        ci_lowers = [results[method][s]["ci_lower"] for s in ensemble_sizes]
        ci_uppers = [results[method][s]["ci_upper"] for s in ensemble_sizes]

        lower_errors = [mean_f1s[i] - ci_lowers[i] for i in range(len(ensemble_sizes))]
        upper_errors = [ci_uppers[i] - mean_f1s[i] for i in range(len(ensemble_sizes))]

        plt.errorbar(
            ensemble_sizes,
            mean_f1s,
            yerr=[lower_errors, upper_errors],
            fmt="o-",
            capsize=5,
            capthick=2,
            linewidth=2,
            markersize=8,
            label=f"{method} (95% CI)",
            color=colors[method_idx],
        )

    plt.xlabel("Number of Ensemble Members", fontsize=12)
    plt.ylabel("F1 Score", fontsize=12)
    plt.title(
        f"Ensemble Size vs F1 Score\n"
        f"(N={num_runs}, reps={cfg.num_repetitions}, "
        f"replacement={cfg.sample_with_replacement}, agg={cfg.aggregation_methods})",
        fontsize=14,
    )
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()

    plot_path = output_dir / "ensemble_size_vs_f1.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved plot to {plot_path}")

    summary_results = {
        "config": {
            "num_runs": num_runs,
            "num_repetitions": cfg.num_repetitions,
            "sample_with_replacement": cfg.sample_with_replacement,
            "aggregation_methods": cfg.aggregation_methods,
            "member_run_dirs": [str(d) for d in member_dirs],
            "gt_dir": str(gt_dir),
        },
        "results": results,
    }

    summary_path = output_dir / "summary_results.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=4)
    logger.info(f"Saved summary results to {summary_path}")

    # Save raw F1 scores
    raw_results = {
        "config": summary_results["config"],
        "raw_f1_scores": {
            method: {str(k): v for k, v in scores.items()}
            for method, scores in all_f1_scores.items()
        },
    }

    raw_path = output_dir / "raw_f1_scores.json"
    with raw_path.open("w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=4)
    logger.info(f"Saved raw F1 scores to {raw_path}")

    logger.info(f"\n{'='*60}")
    logger.info("Experiment complete!")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    CLI(main)
