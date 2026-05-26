#!/usr/bin/env python3
"""
Compare different adaptive strategies.

Runs multiple strategies on the same dataset and compares:
- Accuracy (F1 score)
- Efficiency (tokens, samples, time)
- Cost-effectiveness (F1 per token)
"""

import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from jsonargparse import CLI

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.adaptive.config import ComparisonConfig

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s %(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_strategy(
    strategy_cfg: Dict[str, Any],
    input_images_dir: str,
    output_dir: Path,
) -> Path:
    """Run a single strategy and return output directory."""
    
    strategy_name = strategy_cfg.get("name", strategy_cfg["strategy"])
    logger.info(f"\n{'='*60}")
    logger.info(f"Running strategy: {strategy_name}")
    logger.info(f"{'='*60}")
    
    # Build config for predict.py
    cfg_dict = {
        "input_images_dir": input_images_dir,
        "run_dir": str(output_dir / strategy_name),
        **strategy_cfg,
    }
    
    # Remove 'name' field if present (not part of AdaptivePredictConfig)
    cfg_dict.pop("name", None)
    
    # Save temporary config
    temp_cfg_path = output_dir / f"temp_{strategy_name}.yaml"
    with temp_cfg_path.open("w") as f:
        yaml.safe_dump({"cfg": cfg_dict}, f)
    
    # Run predict script
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "adaptive" / "predict.py"),
        "--config", str(temp_cfg_path),
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    elapsed_time = time.time() - start_time
    
    # Clean up temp config
    temp_cfg_path.unlink()
    
    run_dir = output_dir / strategy_name
    
    # Save timing info
    timing_path = run_dir / "timing.json"
    with timing_path.open("w") as f:
        json.dump({"elapsed_seconds": elapsed_time}, f)
    
    logger.info(f"Completed in {elapsed_time:.1f}s")
    
    return run_dir


def evaluate_strategy(run_dir: Path, gt_dir: str) -> Dict[str, float]:
    """Evaluate a strategy's predictions."""
    
    logger.info(f"Evaluating {run_dir.name}...")
    
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "eval.py"),
        "--cfg.run_dir", str(run_dir),
        "--cfg.gt_dir", gt_dir,
    ]
    
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    # Load eval results
    eval_path = run_dir / "eval_results.json"
    with eval_path.open("r") as f:
        eval_results = json.load(f)
    
    return eval_results["metrics"]


def collect_results(run_dirs: List[Path], gt_dir: str) -> pd.DataFrame:
    """Collect results from all strategy runs."""
    
    results = []
    
    for run_dir in run_dirs:
        # Load statistics
        stats_path = run_dir / "statistics.json"
        with stats_path.open("r") as f:
            stats = json.load(f)
        
        # Load timing
        timing_path = run_dir / "timing.json"
        with timing_path.open("r") as f:
            timing = json.load(f)
        
        # Evaluate
        metrics = evaluate_strategy(run_dir, gt_dir)
        
        # Load config to get strategy details
        cfg_path = run_dir / "config.yaml"
        with cfg_path.open("r") as f:
            cfg = yaml.safe_load(f)
        
        # Compute efficiency metrics
        total_tokens = sum(
            pred["input_tokens"] + pred["output_tokens"]
            for pred in json.load((run_dir / "predictions.json").open())
        )
        
        f1_score = metrics["f1"]
        tokens_per_image = total_tokens / stats["total_images"]
        f1_per_1k_tokens = (f1_score / total_tokens) * 1000 if total_tokens > 0 else 0
        
        results.append({
            "strategy": run_dir.name,
            "f1_score": f1_score,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "avg_samples": stats["avg_samples_per_image"],
            "convergence_rate": stats.get("convergence_rate", 0.0),
            "total_tokens": total_tokens,
            "tokens_per_image": tokens_per_image,
            "elapsed_time": timing["elapsed_seconds"],
            "f1_per_1k_tokens": f1_per_1k_tokens,
            "strategy_type": cfg.get("strategy", "unknown"),
        })
    
    return pd.DataFrame(results)


def plot_comparison(df: pd.DataFrame, output_dir: Path):
    """Create comparison plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Strategy Comparison", fontsize=16, fontweight="bold")
    
    # Plot 1: F1 Score
    ax1 = axes[0, 0]
    df.plot(
        x="strategy",
        y="f1_score",
        kind="bar",
        ax=ax1,
        legend=False,
        color="steelblue",
    )
    ax1.set_title("F1 Score by Strategy")
    ax1.set_ylabel("F1 Score")
    ax1.set_xlabel("")
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Efficiency (F1 per 1K tokens)
    ax2 = axes[0, 1]
    df.plot(
        x="strategy",
        y="f1_per_1k_tokens",
        kind="bar",
        ax=ax2,
        legend=False,
        color="seagreen",
    )
    ax2.set_title("Efficiency (F1 per 1K tokens)")
    ax2.set_ylabel("F1 per 1K tokens")
    ax2.set_xlabel("")
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Average Samples
    ax3 = axes[1, 0]
    df.plot(
        x="strategy",
        y="avg_samples",
        kind="bar",
        ax=ax3,
        legend=False,
        color="coral",
    )
    ax3.set_title("Average Samples per Image")
    ax3.set_ylabel("Avg Samples")
    ax3.set_xlabel("")
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Tokens per Image
    ax4 = axes[1, 1]
    df.plot(
        x="strategy",
        y="tokens_per_image",
        kind="bar",
        ax=ax4,
        legend=False,
        color="mediumpurple",
    )
    ax4.set_title("Tokens per Image")
    ax4.set_ylabel("Tokens")
    ax4.set_xlabel("")
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    plot_path = output_dir / "comparison_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to {plot_path}")
    plt.close()


def main(cfg: ComparisonConfig) -> None:
    """Run strategy comparison."""
    
    logger.info("="*60)
    logger.info("Adaptive Strategy Comparison")
    logger.info("="*60)
    logger.info(f"Input: {cfg.input_images_dir}")
    logger.info(f"Strategies: {len(cfg.strategies)}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = Path(cfg.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output: {output_dir}")
    
    # Limit images if specified
    if cfg.max_images:
        logger.info(f"Limiting to {cfg.max_images} images for testing")
        # TODO: Implement image limiting
    
    # Run each strategy
    run_dirs = []
    for strategy_cfg in cfg.strategies:
        try:
            run_dir = run_strategy(strategy_cfg, cfg.input_images_dir, output_dir)
            run_dirs.append(run_dir)
        except Exception as e:
            logger.error(f"Error running strategy {strategy_cfg}: {e}", exc_info=True)
    
    # Collect and compare results
    logger.info("\n" + "="*60)
    logger.info("Collecting results...")
    logger.info("="*60)
    
    results_df = collect_results(run_dirs, cfg.gt_dir)
    
    # Save results table
    results_path = output_dir / "comparison_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved results to {results_path}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Results Summary")
    logger.info("="*60)
    
    # Sort by F1 score
    results_df_sorted = results_df.sort_values("f1_score", ascending=False)
    
    print("\n" + results_df_sorted.to_string(index=False))
    
    # Find best strategies
    best_accuracy = results_df_sorted.iloc[0]
    best_efficiency = results_df.sort_values("f1_per_1k_tokens", ascending=False).iloc[0]
    
    logger.info("\n" + "="*60)
    logger.info("Best Strategies")
    logger.info("="*60)
    logger.info(f"Best Accuracy: {best_accuracy['strategy']} (F1={best_accuracy['f1_score']:.4f})")
    logger.info(
        f"Best Efficiency: {best_efficiency['strategy']} "
        f"(F1/1K tokens={best_efficiency['f1_per_1k_tokens']:.6f})"
    )
    
    # Create plots
    plot_comparison(results_df, output_dir)
    
    # Save summary
    summary = {
        "timestamp": timestamp,
        "input_images_dir": cfg.input_images_dir,
        "gt_dir": cfg.gt_dir,
        "num_strategies": len(cfg.strategies),
        "best_accuracy": {
            "strategy": best_accuracy["strategy"],
            "f1_score": float(best_accuracy["f1_score"]),
        },
        "best_efficiency": {
            "strategy": best_efficiency["strategy"],
            "f1_per_1k_tokens": float(best_efficiency["f1_per_1k_tokens"]),
        },
    }
    
    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=4)
    
    logger.info(f"\nComparison complete! Results saved to {output_dir}")


if __name__ == "__main__":
    CLI(main)
