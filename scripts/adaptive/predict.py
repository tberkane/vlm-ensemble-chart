#!/usr/bin/env python3
"""
Adaptive prediction script.

Runs adaptive ensemble extraction using configurable sampling strategies.
This is a parallel implementation to scripts/predict.py
"""

import csv
import io
import json
import logging
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from jsonargparse import CLI
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.adaptive import (
    AdaptiveSamplingStrategy,
    DiversitySamplingStrategy,
    ExactMatchConvergence,
    FixedSamplingStrategy,
    HuberAggregation,
    IncrementalEnsembleSamplingStrategy,
    IterationSnapshot,
    MeanAggregation,
    MedianAggregation,
    MedoidAggregation,
    StructuredSamplingStrategy,
    VarianceConvergence,
)
from src.adaptive.aggregation import WeightedConfidenceAggregation, RANSACAggregation
from src.adaptive.config import AdaptivePredictConfig, ConvergenceConfig
from src.eval_chart2table import (
    chart2table_evaluator,
    table_datapoints_precision_recall_per_point,
)
from src.utils import normalize_tsv_table

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s %(asctime)s %(module)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)

# Suppress HTTP request/response logs from API clients
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def load_ground_truth_tables(gt_dir: Path) -> Dict[str, str]:
    """Load ground truth tables from CSV files and convert to TSV."""
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

    logger.info(f"Loaded {len(tables_dict)} ground truth tables from {gt_dir}")
    return tables_dict


import re
from collections import defaultdict
from typing import Any, Dict, List, Optional


def compute_iteration_metrics(
    predictions: List[Dict[str, Any]],
    ground_truth: Dict[str, str],
    *,
    max_iteration: Optional[int] = None,
    patience: int = 2,
) -> Dict[int, Dict[str, float]]:
    """
    Compute F1 and convergence rate at each iteration level.

    Convergence-rate rule (generalized with patience):
      An example is counted as converged at iteration t ONLY if:
        - snapshot at t has converged=True, AND
        - snapshots at (t - (patience-1)), ..., (t-1) all EXIST, AND
        - each of those snapshots has converged=True.

      In other words, the last `patience` consecutive iterations ending at t must all be converged.

    Carry-forward:
      We still carry forward the final TSV for inclusion in later iterations if the
      last snapshot is converged, BUT we only count it as converged in later
      iterations if the last snapshot is "confirmed" converged under the rule above
      (i.e., last `patience` iterations ending at last_iter are all converged).
    """
    if patience < 1:
        raise ValueError(f"patience must be >= 1, got {patience}")

    def _is_confirmed_converged_at(
        iter_num: int, conv_by_iter: Dict[int, bool]
    ) -> bool:
        """
        True iff iterations [iter_num - (patience-1), ..., iter_num] all exist in conv_by_iter
        and are True.
        """
        start = iter_num - (patience - 1)
        if start < 1:
            return False
        for k in range(start, iter_num + 1):
            if not conv_by_iter.get(k, False):
                return False
        return True

    # --- 1) Determine the max iteration we will report over ---
    observed_max_iter = 0
    for pred in predictions:
        image_id = pred.get("image", "").replace(".png", "")
        if image_id not in ground_truth:
            continue
        hist = pred.get("iteration_history") or []
        if not hist:
            continue
        observed_max_iter = max(observed_max_iter, max(s["iteration"] for s in hist))

    if observed_max_iter == 0:
        return {}

    max_iter = (
        min(observed_max_iter, max_iteration)
        if max_iteration is not None
        else observed_max_iter
    )

    # --- 2) Bucket data by iteration ---
    iteration_data = defaultdict(
        lambda: {"predictions": [], "gt": [], "converged_count": 0}
    )

    for pred in predictions:
        image_id = pred.get("image", "").replace(".png", "")
        if image_id not in ground_truth:
            continue

        iteration_history = pred.get("iteration_history") or []
        if not iteration_history:
            continue

        # Sort snapshots to be safe
        iteration_history = sorted(iteration_history, key=lambda s: s["iteration"])
        gt_tsv = ground_truth[image_id]

        # Build quick lookup: iteration -> converged flag
        conv_by_iter = {
            s["iteration"]: bool(s.get("converged", False)) for s in iteration_history
        }

        # Add the real snapshots
        for snapshot in iteration_history:
            iter_num = snapshot["iteration"]
            if iter_num > max_iter:
                break

            iter_tsv = snapshot["tsv_data"]
            iter_tsv = normalize_tsv_table(iter_tsv)
            iter_tsv = re.sub(r"[ ]{2,}", "\t", iter_tsv)

            iteration_data[iter_num]["predictions"].append(iter_tsv)
            iteration_data[iter_num]["gt"].append([f"title\t\n{gt_tsv}"])

            # Patience-based convergence counting
            if _is_confirmed_converged_at(iter_num, conv_by_iter):
                iteration_data[iter_num]["converged_count"] += 1

        # --- 3) Carry-forward if the last snapshot converged and history stopped there ---
        last_snapshot = iteration_history[-1]
        last_iter = last_snapshot["iteration"]
        last_is_conv = bool(last_snapshot.get("converged", False))

        # "Confirmed" converged at last_iter under the patience rule
        last_confirmed = last_is_conv and _is_confirmed_converged_at(
            last_iter, conv_by_iter
        )

        if last_iter < max_iter and last_is_conv:
            carried_tsv_raw = last_snapshot["tsv_data"]
            for iter_num in range(last_iter + 1, max_iter + 1):
                carried_tsv = normalize_tsv_table(carried_tsv_raw)
                carried_tsv = re.sub(r"[ ]{2,}", "\t", carried_tsv)

                iteration_data[iter_num]["predictions"].append(carried_tsv)
                iteration_data[iter_num]["gt"].append([f"title\t\n{gt_tsv}"])

                # Only count carried-forward as converged if we were already "confirmed" at last_iter.
                if last_confirmed:
                    iteration_data[iter_num]["converged_count"] += 1

    # --- 4) Compute metrics for each iteration ---
    iteration_metrics: Dict[int, Dict[str, float]] = {}
    for iter_num in range(1, max_iter + 1):
        data = iteration_data.get(iter_num)
        if not data or not data["predictions"]:
            continue

        predictions_formatted = [f"title\t\n{p}" for p in data["predictions"]]

        # Compute per-point metrics (this is what table_datapoints_precision_recall does internally anyway)
        per_point = table_datapoints_precision_recall_per_point(
            data["gt"], predictions_formatted
        )

        # Compute average F1 from per-point scores
        f1_list = per_point.get("f1", [])
        avg_f1 = (sum(f1_list) / len(f1_list)) if f1_list else 0.0

        # Compute aggregated breakdown by averaging per-point breakdowns
        n = len(f1_list) if f1_list else 1
        correct = sum(per_point.get("correct", [])) / n
        value_errors = sum(per_point.get("value_errors", [])) / n
        label_errors = sum(per_point.get("label_errors", [])) / n
        missing_datapoints = sum(per_point.get("missing_datapoints", [])) / n
        extra_datapoints = sum(per_point.get("extra_datapoints", [])) / n

        # Pin "correct" to aggregated F1 (matching table_datapoints_precision_recall behavior)
        correct = avg_f1 * 100.0
        total = (
            correct
            + value_errors
            + label_errors
            + missing_datapoints
            + extra_datapoints
        )
        extra_datapoints += 100.0 - total

        total_images = len(data["predictions"])
        convergence_rate = (
            (data["converged_count"] / total_images) if total_images > 0 else 0.0
        )

        iteration_metrics[iter_num] = {
            "f1": avg_f1 * 100.0,
            "convergence_rate": convergence_rate * 100.0,
            "num_images": float(total_images),
            "breakdown": {
                "correct": correct,
                "value_errors": value_errors,
                "label_errors": label_errors,
                "missing_datapoints": missing_datapoints,
                "extra_datapoints": extra_datapoints,
            },
        }

    return iteration_metrics


def create_convergence_checker(cfg: ConvergenceConfig):
    """Create convergence checker from config."""
    if cfg.type == "exact_match":
        return ExactMatchConvergence(num_matches=cfg.num_matches)

    elif cfg.type == "variance":
        return VarianceConvergence(
            threshold=cfg.variance_threshold,
            require_all_cells=cfg.require_all_cells,
            min_samples=cfg.min_samples,
            patience=cfg.patience,
        )

    else:
        raise ValueError(f"Unknown convergence type: {cfg.type}")


def create_strategy(cfg: AdaptivePredictConfig):
    """Create sampling strategy from config."""
    # Create aggregation method
    if cfg.aggregation == "median":
        aggregation = MedianAggregation(
            row_sim_threshold=cfg.row_similarity_threshold,
            col_sim_threshold=cfg.column_similarity_threshold,
            pruning=cfg.pruning,
            pruning_threshold=cfg.pruning_threshold,
        )
    elif cfg.aggregation == "medoid":
        aggregation = MedoidAggregation(
            row_sim_threshold=cfg.row_similarity_threshold,
            col_sim_threshold=cfg.column_similarity_threshold,
            pruning=cfg.pruning,
            pruning_threshold=cfg.pruning_threshold,
        )
    elif cfg.aggregation == "mean":
        aggregation = MeanAggregation(
            row_sim_threshold=cfg.row_similarity_threshold,
            col_sim_threshold=cfg.column_similarity_threshold,
            pruning=cfg.pruning,
            pruning_threshold=cfg.pruning_threshold,
        )
    elif cfg.aggregation == "huber":
        aggregation = HuberAggregation(
            row_sim_threshold=cfg.row_similarity_threshold,
            col_sim_threshold=cfg.column_similarity_threshold,
            pruning=cfg.pruning,
            pruning_threshold=cfg.pruning_threshold,
        )
    elif cfg.aggregation == "weighted_confidence":
        aggregation = WeightedConfidenceAggregation(
            row_sim_threshold=cfg.row_similarity_threshold,
            col_sim_threshold=cfg.column_similarity_threshold,
            pruning=cfg.pruning,
            pruning_threshold=cfg.pruning_threshold,
        )
    elif cfg.aggregation == "ransac":
        aggregation = RANSACAggregation(
            row_sim_threshold=cfg.row_similarity_threshold,
            col_sim_threshold=cfg.column_similarity_threshold,
            pruning=cfg.pruning,
            pruning_threshold=cfg.pruning_threshold,
        )
    else:
        raise ValueError(f"Unknown aggregation: {cfg.aggregation}")

    # Create strategy
    if cfg.strategy == "fixed":
        return FixedSamplingStrategy(
            num_samples=cfg.fixed_num_samples,
            model=cfg.model,
            temperature=cfg.temperature,
            aggregation=aggregation,
        )

    elif cfg.strategy == "adaptive":
        convergence_checker = create_convergence_checker(cfg.convergence)
        return AdaptiveSamplingStrategy(
            min_samples=cfg.adaptive_min_samples,
            max_samples=cfg.adaptive_max_samples,
            model=cfg.model,
            convergence_checker=convergence_checker,
            temperature=cfg.temperature,
            aggregation=aggregation,
        )

    elif cfg.strategy == "diversity":
        models = cfg.diversity_models or [cfg.model]
        prompt_variants = cfg.diversity_prompt_variants or ["standard"]

        return DiversitySamplingStrategy(
            models=models,
            prompt_variants=prompt_variants,
            samples_per_config=cfg.diversity_samples_per_config,
            temperature=cfg.temperature,
            aggregation=aggregation,
        )

    elif cfg.strategy == "incremental_ensemble":
        return IncrementalEnsembleSamplingStrategy(
            initial_batch_size=cfg.incremental_initial_batch_size,
            max_samples=cfg.incremental_max_samples,
            model=cfg.model,
            temperature=cfg.temperature,
            aggregation=aggregation,
            patience=cfg.incremental_patience,
            exact_match=cfg.incremental_exact_match,
            convergence_coverage=cfg.incremental_convergence_coverage,
            convergence_tolerance=cfg.incremental_convergence_tolerance,
        )

    elif cfg.strategy == "structured_sampling":
        return StructuredSamplingStrategy(
            initial_batch_size=cfg.incremental_initial_batch_size,
            max_samples=cfg.incremental_max_samples,
            model=cfg.model,
            structure_model=cfg.structure_model,
            temperature=cfg.temperature,
            aggregation=aggregation,
            patience=cfg.incremental_patience,
        )
    else:
        raise ValueError(f"Unknown strategy: {cfg.strategy}")


def main(cfg: AdaptivePredictConfig) -> None:
    """Run adaptive prediction."""
    load_dotenv()

    logger.info("=" * 60)
    logger.info("Adaptive Ensemble Prediction")
    logger.info("=" * 60)
    logger.info(f"Strategy: {cfg.strategy}")
    logger.info(f"Model: {cfg.model}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    input_dir_name = "_".join(
        str(Path(cfg.input_images_dir).resolve()).split("/")[-3:-1]
    ).replace(" ", "_")
    tag = (
        f"{cfg.model.replace('/', '_')}_"
        f"temp{str(cfg.temperature).replace('.', 'p')}_"
        f"maxsamp{cfg.incremental_max_samples}_"
        f"pat{cfg.incremental_patience}_"
        f"{cfg.aggregation}"
    )
    run_name = f"{timestamp}__{input_dir_name}__{tag}"

    if cfg.run_dir:
        run_dir = Path(cfg.run_dir)
    else:
        run_dir = project_root / "outputs" / "adaptive" / run_name

    # Ensure directory exists
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {run_dir}")
        if not run_dir.exists():
            raise RuntimeError(f"Failed to create directory: {run_dir}")
    except Exception as e:
        logger.error(f"Error creating output directory {run_dir}: {e}")
        raise

    # Find images
    images_dir = Path(cfg.input_images_dir)
    if not images_dir.is_absolute():
        images_dir = project_root / images_dir

    image_files = sorted(images_dir.glob("*.png"))
    logger.info(f"Found {len(image_files)} images")

    # Create strategy
    strategy = create_strategy(cfg)
    logger.info(f"Strategy config: {strategy.get_config()}")

    # Run predictions
    predictions: List[Dict[str, Any]] = []
    convergence_stats = {
        "converged": 0,
        "max_samples": 0,
        "total_samples": 0,
        "convergence_iterations": [],
        "images_with_fuzzy_failures": 0,  # Images that had at least one fuzzy match failure
        "total_fuzzy_failures": 0,  # Total number of fuzzy match failures across all images
    }

    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            result = strategy.extract(image_path)

            # Extract fuzzy match failures from metadata
            fuzzy_failures = (
                result.metadata.get("fuzzy_match_failures", 0) if result.metadata else 0
            )

            # Convert iteration_history to serializable format
            iteration_history_dict = None
            if result.iteration_history:
                iteration_history_dict = [
                    {
                        "iteration": snapshot.iteration,
                        "tsv_data": snapshot.tsv_data,
                        "converged": snapshot.converged,
                    }
                    for snapshot in result.iteration_history
                ]

            predictions.append(
                {
                    "image": image_path.name,
                    "answer": result.tsv_data,
                    "mad_tsv": result.mad_tsv,  # MAD table in TSV format
                    "input_tokens": result.total_input_tokens,
                    "output_tokens": result.total_output_tokens,
                    "num_samples": result.num_samples,
                    "converged": result.converged,
                    "convergence_iteration": result.convergence_iteration,
                    "fuzzy_match_failures": fuzzy_failures,
                    "iteration_history": iteration_history_dict,
                }
            )

            # Update stats
            convergence_stats["total_samples"] += result.num_samples
            if result.converged:
                convergence_stats["converged"] += 1
                convergence_stats["convergence_iterations"].append(
                    result.convergence_iteration
                )
            else:
                convergence_stats["max_samples"] += 1

            # Track fuzzy matching failures
            if fuzzy_failures > 0:
                convergence_stats["images_with_fuzzy_failures"] += 1
                convergence_stats["total_fuzzy_failures"] += fuzzy_failures

        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {e}", exc_info=True)
            predictions.append(
                {
                    "image": image_path.name,
                    "answer": "",
                    "mad_tsv": None,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "num_samples": 0,
                    "converged": False,
                    "convergence_iteration": None,
                    "fuzzy_match_failures": 0,
                    "error": str(e),
                }
            )

    # Save predictions
    # Ensure directory still exists before writing
    run_dir.mkdir(parents=True, exist_ok=True)

    pred_path = run_dir / "predictions.json"
    with pred_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=4)
    logger.info(f"Saved predictions to {pred_path}")

    # Compute and save statistics
    avg_samples = convergence_stats["total_samples"] / len(predictions)
    convergence_rate = (
        convergence_stats["converged"] / len(predictions) if predictions else 0
    )

    if convergence_stats["convergence_iterations"]:
        avg_convergence_iter = sum(convergence_stats["convergence_iterations"]) / len(
            convergence_stats["convergence_iterations"]
        )
    else:
        avg_convergence_iter = None

    # Compute fuzzy matching failure stats
    fuzzy_failure_rate = (
        convergence_stats["images_with_fuzzy_failures"] / len(predictions)
        if predictions
        else 0
    )
    avg_fuzzy_failures = (
        convergence_stats["total_fuzzy_failures"]
        / convergence_stats["images_with_fuzzy_failures"]
        if convergence_stats["images_with_fuzzy_failures"] > 0
        else 0
    )

    stats = {
        "total_images": len(predictions),
        "converged": convergence_stats["converged"],
        "max_samples_reached": convergence_stats["max_samples"],
        "convergence_rate": convergence_rate,
        "avg_samples_per_image": avg_samples,
        "avg_convergence_iteration": avg_convergence_iter,
        "total_samples": convergence_stats["total_samples"],
        "images_with_fuzzy_failures": convergence_stats["images_with_fuzzy_failures"],
        "fuzzy_failure_rate": fuzzy_failure_rate,
        "total_fuzzy_failures": convergence_stats["total_fuzzy_failures"],
        "avg_fuzzy_failures_per_affected_image": avg_fuzzy_failures,
    }

    stats_path = run_dir / "statistics.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)

    logger.info("\n" + "=" * 60)
    logger.info("Statistics:")
    logger.info(f"  Convergence rate: {convergence_rate*100:.1f}%")
    logger.info(f"  Avg samples per image: {avg_samples:.2f}")
    if avg_convergence_iter:
        logger.info(f"  Avg convergence iteration: {avg_convergence_iter:.2f}")
    logger.info(
        f"  Images with fuzzy match failures: {convergence_stats['images_with_fuzzy_failures']} ({fuzzy_failure_rate*100:.1f}%)"
    )
    logger.info(
        f"  Total fuzzy match failures: {convergence_stats['total_fuzzy_failures']}"
    )
    if avg_fuzzy_failures > 0:
        logger.info(
            f"  Avg fuzzy failures per affected image: {avg_fuzzy_failures:.2f}"
        )
    logger.info("=" * 60)

    # Save config
    cfg_dict = vars(cfg).copy()
    cfg_dict["convergence"] = vars(cfg.convergence)
    cfg_dict["strategy_config"] = strategy.get_config()

    cfg_path = run_dir / "config.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Saved config to {cfg_path}")

    # Compute iteration-level metrics if ground truth is available
    if cfg.gt_dir:
        logger.info("\n" + "=" * 60)
        logger.info("Computing iteration-level metrics...")
        logger.info("=" * 60)

        try:
            gt_dir = Path(cfg.gt_dir)
            if not gt_dir.is_absolute():
                gt_dir = project_root / gt_dir

            ground_truth = load_ground_truth_tables(gt_dir)
            iteration_metrics = compute_iteration_metrics(
                predictions, ground_truth, patience=cfg.incremental_patience
            )

            # Save iteration metrics
            iteration_metrics_path = run_dir / "iteration_metrics.json"
            with iteration_metrics_path.open("w", encoding="utf-8") as f:
                # Convert dict keys to strings for JSON serialization
                serializable_metrics = {str(k): v for k, v in iteration_metrics.items()}
                json.dump(serializable_metrics, f, indent=4)

            logger.info(f"Saved iteration metrics to {iteration_metrics_path}")

            # Log summary
            logger.info("\nIteration-level metrics computed:")
            for iter_num in sorted(iteration_metrics.keys()):
                metrics = iteration_metrics[iter_num]
                breakdown = metrics.get("breakdown", {})
                logger.info(
                    f"  Iteration {iter_num}: F1={metrics['f1']:.2f}%, "
                    f"Convergence={metrics['convergence_rate']:.1f}%, "
                    f"N={metrics['num_images']}"
                )
                if breakdown:
                    logger.info(
                        f"    Breakdown: correct={breakdown.get('correct', 0):.2f}%, "
                        f"value_errors={breakdown.get('value_errors', 0):.2f}%, "
                        f"label_errors={breakdown.get('label_errors', 0):.2f}%, "
                        f"missing={breakdown.get('missing_datapoints', 0):.2f}%, "
                        f"extra={breakdown.get('extra_datapoints', 0):.2f}%"
                    )
        except Exception as e:
            logger.error(f"Failed to compute iteration metrics: {e}", exc_info=True)


if __name__ == "__main__":
    CLI(main)
