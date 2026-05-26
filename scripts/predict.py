# Script to generate predictions for all chart images in a given directory.

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import wandb
import yaml
from dotenv import load_dotenv
from jsonargparse import CLI
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.config import PredictConfig
from src.extract_data import (
    CHART_EXTRACTION_PROMPT,
    CHART_EXTRACTION_PROMPT_WB,
    extract_data_from_chart,
)

logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_run_dir(cfg: PredictConfig) -> Tuple[Path, str]:
    """Create a unique directory for this prediction run and save the config."""
    input_path = Path(cfg.input_images_dir)
    parts = input_path.parts
    if "data" in parts:
        data_idx = parts.index("data")
        after_data = parts[data_idx + 1 : data_idx + 3]
        if len(after_data) == 2:
            dataset_name = f"{after_data[0]}_{after_data[1]}"
        elif len(after_data) == 1:
            dataset_name = after_data[0]
        else:
            dataset_name = "data"
    else:
        dataset_name = (
            "_".join(parts[:2]) if len(parts) >= 2 else (parts[0] if parts else "data")
        )
    dataset_name = dataset_name.replace(" ", "_").replace("/", "_")

    if cfg.temperature is None:
        temp_str = ""
    else:
        temp_str = f"_temp{cfg.temperature}".replace(".", "p")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    short_tag = f"{dataset_name}_{cfg.model.split('/')[-1]}{temp_str}_{cfg.prompt}"
    run_name = f"{timestamp}__{short_tag}"

    runs_root = project_root / "outputs" / "predict"
    run_dir = runs_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    # Save resolved config
    config_path = run_dir / "config.yaml"
    with config_path.open("w") as f:
        yaml.safe_dump(vars(cfg), f)

    logger.info(f"Run directory created at: {run_dir}")
    return run_dir, run_name


def load_existing_predictions(output_path: Path) -> List[Dict[str, Any]]:
    """Load existing predictions if the file exists, otherwise return an empty list."""
    if not output_path.exists():
        return []

    with output_path.open("r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Could not decode JSON from {output_path}, starting fresh.")
            return []


def main(cfg: PredictConfig):
    load_dotenv()

    if cfg.run_dir:
        run_dir = Path(cfg.run_dir)
        if not run_dir.is_absolute():
            run_dir = project_root / run_dir

        run_dir.mkdir(parents=True, exist_ok=True)
        run_name = run_dir.name
        logger.info(f"Resuming from existing run directory: {run_dir}")
    else:
        run_dir, run_name = create_run_dir(cfg)

    if cfg.prompt == "CHART_EXTRACTION_PROMPT":
        prompt = CHART_EXTRACTION_PROMPT
    elif cfg.prompt == "CHART_EXTRACTION_PROMPT_WB":
        prompt = CHART_EXTRACTION_PROMPT_WB
    else:
        raise ValueError(f"Invalid prompt: {cfg.prompt}")

    # Init Weights & Biases
    wandb_run = None
    wandb_meta_path = run_dir / "wandb_run.json"

    if cfg.use_wandb:
        # Case 1: resuming and prior W&B run exists
        if cfg.run_dir and wandb_meta_path.exists():
            with wandb_meta_path.open("r") as f:
                meta = json.load(f)
            wandb_run = wandb.init(
                project=cfg.wandb_project,
                id=meta["id"],
                resume="allow",
                dir=str(run_dir),
            )
            logger.info(
                f"Resuming W&B run: {wandb_run.name} (id={wandb_run.id}) "
                f"from {wandb_meta_path}"
            )
        else:
            # Case 2: new W&B run (or run_dir had no W&B before)
            wandb_run = wandb.init(
                project=cfg.wandb_project,
                name=run_name,
                dir=str(run_dir),
                config=vars(cfg),
            )
            logger.info(f"Started new W&B run: {wandb_run.name} ({wandb_run.id})")

            # Save run metadata so we can resume later
            with wandb_meta_path.open("w") as f:
                json.dump({"id": wandb_run.id, "name": wandb_run.name}, f)

    image_dir = project_root / cfg.input_images_dir
    output_path = run_dir / "predictions.json"

    if not image_dir.exists():
        raise FileNotFoundError(f"Input images directory does not exist: {image_dir}")

    all_images = sorted(
        [
            f
            for f in os.listdir(image_dir)
            if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg")
        ]
    )

    logger.info(f"Found {len(all_images)} images in {image_dir}")

    answer_dict: List[Dict[str, Any]] = load_existing_predictions(output_path)
    already_done = set(entry["image"] for entry in answer_dict)

    total_input_tokens = sum(entry.get("input_tokens", 0) for entry in answer_dict)
    total_output_tokens = sum(entry.get("output_tokens", 0) for entry in answer_dict)
    num_failed = sum(1 for entry in answer_dict if not entry.get("answer", "").strip())
    processed_count = len(answer_dict)

    # Exponential backoff parameters
    base_delay = 2.0
    current_delay = base_delay
    first_new_image = True  # Track if this is the first new image being processed

    for image_name in tqdm(all_images, desc="Processing images"):
        if image_name in already_done:
            continue

        # Apply exponential backoff before making the API call
        if not first_new_image:  # Skip delay for the first new image
            time.sleep(current_delay)
            logger.debug(
                f"Applied backoff delay of {current_delay:.2f}s before processing {image_name}"
            )
        first_new_image = False

        image_path = image_dir / image_name

        max_attempts = 6
        pred = None
        res: Dict[str, Any] = {}

        for _ in range(max_attempts):
            try:
                res = extract_data_from_chart(
                    image_path,
                    cfg.model,
                    temperature=cfg.temperature,
                    prompt=prompt,
                )
            except Exception as e:
                logger.error(f"Error in extract_data_from_chart for {image_name}: {e}")
                res = {"csv_data": None, "input_tokens": 0, "output_tokens": 0}

            pred = res.get("csv_data") or ""
            if pred:
                break

        # Update exponential backoff for next call
        if pred:
            # Reset delay on success
            current_delay = base_delay
        else:
            # Increase delay on failure (exponential backoff)
            current_delay = current_delay * 2

        if not pred:
            logger.warning(
                f"No valid prediction after {max_attempts} attempts for {image_name}"
            )
            num_failed += 1
            pred = ""
            res.setdefault("input_tokens", 0)
            res.setdefault("output_tokens", 0)

        input_tokens = res.get("input_tokens", 0)
        output_tokens = res.get("output_tokens", 0)

        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        answer_dict.append(
            {
                "image": image_name,
                "answer": pred,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )

        with output_path.open("w") as f:
            json.dump(answer_dict, f, indent=4)

        # Per-image W&B logging
        if wandb_run is not None:
            processed_count += 1
            wandb.log(
                {
                    "per_image/input_tokens": input_tokens,
                    "per_image/output_tokens": output_tokens,
                    "cumulative/input_tokens": total_input_tokens,
                    "cumulative/output_tokens": total_output_tokens,
                    "num_failed": num_failed,
                },
                step=processed_count,  # Use processed_count instead of idx
            )

    metrics_path = run_dir / "metrics.json"
    summary = {
        "num_images": len(all_images),
        "num_processed": len(answer_dict),
        "num_failed": num_failed,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }
    with metrics_path.open("w") as f:
        json.dump(summary, f, indent=4)

    # W&B summary & finish
    if wandb_run is not None:
        wandb.summary["num_images"] = len(all_images)
        wandb.summary["num_processed"] = len(answer_dict)
        wandb.summary["num_failed"] = num_failed
        wandb.summary["total_input_tokens"] = total_input_tokens
        wandb.summary["total_output_tokens"] = total_output_tokens
        wandb_run.finish()


if __name__ == "__main__":
    CLI(main)
