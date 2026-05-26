import csv
import io
import json
import logging
import re
import sys
from pathlib import Path

import wandb
import yaml
from jsonargparse import CLI

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.config import EvalConfig
from src.eval_chart2table import chart2table_evaluator
from src.utils import normalize_tsv_table

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s %(asctime)s %(module)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_PRICING = {
    "qwen/qwen3-vl-8b-instruct": {
        "input_cost_per_1M": 0.064,
        "output_cost_per_1M": 0.4,
    },
    "qwen/qwen3-vl-235b-a22b-instruct": {
        "input_cost_per_1M": 0.2,
        "output_cost_per_1M": 1.2,
    },
    "meta-llama/llama-4-scout-17b-16e-instruct": {
        "input_cost_per_1M": 0.11,
        "output_cost_per_1M": 0.34,
    },
    "meta-llama/llama-4-maverick-17b-128e-instruct": {
        "input_cost_per_1M": 0.2,
        "output_cost_per_1M": 0.6,
    },
    "gemini-3-pro-preview": {
        "input_cost_per_1M": 2.0,
        "output_cost_per_1M": 12.0,
    },
    "gpt-5.1": {
        "input_cost_per_1M": 1.25,
        "output_cost_per_1M": 10.0,
    },
    "TinyChart": {
        "input_cost_per_1M": 0.0,
        "output_cost_per_1M": 0.0,
    },
    "OneChart": {
        "input_cost_per_1M": 0.0,
        "output_cost_per_1M": 0.0,
    },
    "bytedance-seed/seed-1.6-flash": {
        "input_cost_per_1M": 0.075,
        "output_cost_per_1M": 0.3,
    },
}


def main(cfg: EvalConfig):
    run_dir = Path(cfg.run_dir)
    predictions_path = run_dir / "predictions.json"

    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    # ------------------------------------------------------------------
    # 1. Load predictions
    # ------------------------------------------------------------------
    with predictions_path.open("r", encoding="utf-8") as f:
        predictions = json.load(f)

    predictions_dict = {
        prediction["image"].split(".")[0]: prediction["answer"]
        for prediction in predictions
    }
    logger.info(f"Loaded {len(predictions_dict)} predictions.")

    # ------------------------------------------------------------------
    # 2. Load ground-truth tables
    # ------------------------------------------------------------------
    gt_dir = Path(cfg.gt_dir)
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")

    tables_dict = {}
    for csv_file in gt_dir.glob("*.csv"):
        with csv_file.open("r", encoding="utf-8") as f:
            csv_content = f.read()
            # convert csv to tsv
            reader = csv.reader(io.StringIO(csv_content))
            output = io.StringIO()
            writer = csv.writer(output, delimiter="\t", lineterminator="\n")
            for row in reader:
                writer.writerow(row)
            tsv_content = output.getvalue()
            tsv_content = normalize_tsv_table(tsv_content)
            tables_dict[csv_file.name.split(".")[0]] = tsv_content
    logger.info(f"Loaded {len(tables_dict)} tables from {gt_dir}")

    # ------------------------------------------------------------------
    # 3. Match predictions with ground truth
    # ------------------------------------------------------------------
    preds_and_gts = []
    missing_tables = 0
    matched_pred_indices = []
    for idx, (image, prediction) in enumerate(predictions_dict.items()):
        table = tables_dict.get(image)
        if table is None:
            missing_tables += 1
            continue
        else:
            prediction = normalize_tsv_table(prediction)
            # Replace multiple contiguous spaces with \t
            prediction = re.sub(r"[ ]{2,}", "\t", prediction)

            preds_and_gts.append(
                {
                    "model_answer": prediction,
                    "gt_answer": table,
                    "prediction_index": idx,
                    "image": image,
                }
            )
            matched_pred_indices.append(image)
    if missing_tables > 0:
        logger.warning(
            f"{missing_tables} predictions could not be matched with a ground truth table."
        )

    # ------------------------------------------------------------------
    # 4. Compute F1
    # ------------------------------------------------------------------
    metrics = chart2table_evaluator(preds_and_gts)
    f1 = metrics["table_datapoints_f1"]
    logger.info(f"F1: {f1:.4f}")

    # ------------------------------------------------------------------
    # 5. Aggregate token usage & compute costs
    # ------------------------------------------------------------------
    image_to_tokens = {
        pred["image"].split(".")[0]: (pred["input_tokens"], pred["output_tokens"])
        for pred in predictions
    }
    total_input_tokens = 0
    total_output_tokens = 0
    for matched in preds_and_gts:
        image_key = matched["image"]
        if image_key in image_to_tokens:
            input_tokens, output_tokens = image_to_tokens[image_key]
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

    logger.info(f"Total input tokens: {total_input_tokens}")
    logger.info(f"Total output tokens: {total_output_tokens}")

    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        run_config = yaml.safe_load(f)

    # Check if this is an ensemble run (has member_models instead of model)
    is_ensemble = (
        "member_models" in run_config and run_config["member_models"] is not None
    )

    if is_ensemble:
        # Handle ensemble run: compute costs for each member run separately
        member_run_dirs = run_config.get("member_run_dirs", [])
        if not member_run_dirs:
            raise ValueError(f"member_run_dirs is empty in config.yaml: {config_path}")

        logger.info(f"Ensemble run detected with {len(member_run_dirs)} member run(s)")

        # Calculate costs for each member run
        total_input_cost_usd = 0.0
        total_output_cost_usd = 0.0
        member_costs = []

        for member_dir_str in member_run_dirs:
            # Resolve member run directory path
            member_dir = Path(member_dir_str)
            if not member_dir.is_absolute():
                member_dir = project_root / member_dir

            if not member_dir.exists():
                raise FileNotFoundError(f"Member run dir does not exist: {member_dir}")

            # Load member run config to get model
            member_config_path = member_dir / "config.yaml"
            if not member_config_path.exists():
                raise FileNotFoundError(
                    f"config.yaml not found in member run: {member_config_path}"
                )

            with member_config_path.open("r", encoding="utf-8") as f:
                member_config = yaml.safe_load(f)

            model_name = member_config.get("model")
            if model_name is None:
                raise ValueError(
                    f"Model not found in member run config: {member_config_path}"
                )

            if model_name not in MODEL_PRICING:
                raise ValueError(
                    f"Model '{model_name}' not found in pricing dict for member run {member_dir}. "
                    f"Available models: {list(MODEL_PRICING.keys())}"
                )

            pricing = MODEL_PRICING[model_name]
            input_cost_per_1M = pricing["input_cost_per_1M"]
            output_cost_per_1M = pricing["output_cost_per_1M"]

            # Load member run predictions to get token counts
            member_pred_path = member_dir / "predictions.json"
            if not member_pred_path.exists():
                raise FileNotFoundError(
                    f"predictions.json not found in member run: {member_pred_path}"
                )

            with member_pred_path.open("r", encoding="utf-8") as f:
                member_predictions = json.load(f)

            # Sum tokens for this member run
            member_input_tokens = sum(
                pred.get("input_tokens", 0) for pred in member_predictions
            )
            member_output_tokens = sum(
                pred.get("output_tokens", 0) for pred in member_predictions
            )

            # Calculate cost for this member run
            member_input_cost = (member_input_tokens / 1_000_000.0) * input_cost_per_1M
            member_output_cost = (
                member_output_tokens / 1_000_000.0
            ) * output_cost_per_1M
            member_total_cost = member_input_cost + member_output_cost

            total_input_cost_usd += member_input_cost
            total_output_cost_usd += member_output_cost

            member_costs.append(
                {
                    "member_dir": str(member_dir),
                    "model": model_name,
                    "input_tokens": member_input_tokens,
                    "output_tokens": member_output_tokens,
                    "input_cost_usd": member_input_cost,
                    "output_cost_usd": member_output_cost,
                    "total_cost_usd": member_total_cost,
                }
            )

            logger.info(
                f"Member run {member_dir.name}: {model_name} - "
                f"input={member_input_tokens} tokens (${member_input_cost:.6f}), "
                f"output={member_output_tokens} tokens (${member_output_cost:.6f}), "
                f"total=${member_total_cost:.6f}"
            )

        total_cost_usd = total_input_cost_usd + total_output_cost_usd

        # Calculate total tokens from member runs for weighted average pricing
        total_member_input_tokens = sum(mc["input_tokens"] for mc in member_costs)
        total_member_output_tokens = sum(mc["output_tokens"] for mc in member_costs)

        # For eval_results, we'll use weighted average pricing for display
        # (but the actual cost is the sum computed above)
        if total_member_input_tokens > 0:
            input_cost_per_1M = (
                total_input_cost_usd / total_member_input_tokens
            ) * 1_000_000.0
        else:
            input_cost_per_1M = 0.0

        if total_member_output_tokens > 0:
            output_cost_per_1M = (
                total_output_cost_usd / total_member_output_tokens
            ) * 1_000_000.0
        else:
            output_cost_per_1M = 0.0

        logger.info(
            f"Total ensemble cost: input=${total_input_cost_usd:.6f}, "
            f"output=${total_output_cost_usd:.6f}, total=${total_cost_usd:.6f}"
        )
    else:
        # Handle regular (non-ensemble) run
        model_name = run_config.get("model")
        if model_name is None:
            raise ValueError(f"Model not found in config.yaml: {config_path}")

        if model_name not in MODEL_PRICING:
            raise ValueError(
                f"Model '{model_name}' not found in pricing dict: {MODEL_PRICING.keys()}"
            )

        pricing = MODEL_PRICING[model_name]
        input_cost_per_1M = pricing["input_cost_per_1M"]
        output_cost_per_1M = pricing["output_cost_per_1M"]

        total_input_cost_usd = (total_input_tokens / 1_000_000.0) * input_cost_per_1M
        total_output_cost_usd = (total_output_tokens / 1_000_000.0) * output_cost_per_1M
        total_cost_usd = total_input_cost_usd + total_output_cost_usd

        logger.info(f"Total input cost (USD):  {total_input_cost_usd:.6f}")
        logger.info(f"Total output cost (USD): {total_output_cost_usd:.6f}")
        logger.info(f"Total cost (USD):        {total_cost_usd:.6f}")

    # ------------------------------------------------------------------
    # 6. Save eval_results.json
    # ------------------------------------------------------------------
    eval_results = {
        "run_dir": str(run_dir),
        "gt_dir": str(gt_dir),
        "num_predictions": len(predictions_dict),
        "num_evaluated": len(preds_and_gts),
        "is_ensemble": is_ensemble,
        "metrics": {
            "f1": f1,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "input_cost_per_1M_tokens_usd": input_cost_per_1M,
            "output_cost_per_1M_tokens_usd": output_cost_per_1M,
            "total_input_cost_usd": total_input_cost_usd,
            "total_output_cost_usd": total_output_cost_usd,
            "total_cost_usd": total_cost_usd,
        },
    }

    # Add ensemble-specific metadata if applicable
    if is_ensemble:
        member_models = run_config.get("member_models", {})
        eval_results["ensemble_info"] = {
            "aggregation": run_config.get("aggregation", "unknown"),
            "num_member_runs": len(member_models),
            "member_models": member_models,
            "member_costs": member_costs,
        }
    else:
        eval_results["model"] = run_config.get("model")

    eval_results_path = run_dir / "eval_results.json"
    with eval_results_path.open("w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=4)
    logger.info(f"Saved eval results to {eval_results_path}")

    # ------------------------------------------------------------------
    # 7. Log to Weights & Biases
    # ------------------------------------------------------------------
    use_wandb = getattr(cfg, "use_wandb", False)
    if use_wandb:
        wandb_project = getattr(cfg, "wandb_project", "vlm-ensemble-chart-extraction")
        wandb_run_name = f"eval__{run_dir.name}"

        run = wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config=vars(cfg),
            dir=str(run_dir),
        )

        wandb.summary["f1"] = f1
        wandb.summary["num_predictions"] = len(predictions_dict)
        wandb.summary["num_evaluated"] = len(preds_and_gts)
        wandb.summary["total_input_tokens"] = total_input_tokens
        wandb.summary["total_output_tokens"] = total_output_tokens
        wandb.summary["total_input_cost_usd"] = total_input_cost_usd
        wandb.summary["total_output_cost_usd"] = total_output_cost_usd
        wandb.summary["total_cost_usd"] = total_cost_usd

        run.finish()
        logger.info("Logged eval metrics to Weights & Biases.")


if __name__ == "__main__":
    CLI(main)
