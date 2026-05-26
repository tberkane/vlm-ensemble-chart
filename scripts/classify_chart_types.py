"""
Classify ChartQA test images into chart-type categories using a lightweight VLM.

Reads PNGs from a directory, calls a small Gemini model on each, and writes
{image_name: chart_type} to a JSON file. Caching is per-image so reruns/resumes
are free.

Usage:
    python scripts/classify_chart_types.py \
        --image_dir "data/ChartQA Dataset/test/png" \
        --output outputs/chartqa_chart_types.json \
        --model gemini-3.1-flash-lite \
        --workers 16
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.extract_data import encode_image_base64, get_image_type, get_model_client

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Canonical chart-type labels. Order is the order shown in the paper table.
LABELS = [
    "Vertical bar chart",
    "Horizontal bar chart",
    "Stacked bar chart",
    "Line chart",
    "Area chart",
    "Pie chart",
    "Combo chart",
    "Other",
]

PROMPT = (
    "Classify this chart image into exactly one of the following categories. "
    "Reply with ONLY the category name, exactly as written, and nothing else.\n\n"
    "Categories:\n"
    "- Vertical bar chart  (rectangular bars rising from a horizontal axis; single series or grouped)\n"
    "- Horizontal bar chart  (rectangular bars extending from a vertical axis)\n"
    "- Stacked bar chart  (bars subdivided into stacked segments per category)\n"
    "- Line chart  (one or more lines connecting datapoints over an axis)\n"
    "- Area chart  (filled regions under lines, including stacked areas)\n"
    "- Pie chart  (circular chart divided into slices)\n"
    "- Combo chart  (mix of bars and lines on the same plot)\n"
    "- Other  (anything that does not fit the above)\n"
)

LABEL_LOOKUP = {label.lower(): label for label in LABELS}


def normalize_label(raw: str) -> str:
    """Map a raw model response to one of the canonical labels."""
    if raw is None:
        return "Other"
    text = raw.strip().strip(".").strip('"').strip("'")
    # Take only the first non-empty line
    for line in text.splitlines():
        line = line.strip()
        if line:
            text = line
            break
    key = text.lower()
    if key in LABEL_LOOKUP:
        return LABEL_LOOKUP[key]
    # Fallback: substring match against the canonical labels
    for label_key, label in LABEL_LOOKUP.items():
        if label_key in key:
            return label
    # Heuristic keywords
    if "horizontal" in key and "bar" in key:
        return "Horizontal bar chart"
    if "stacked" in key and ("bar" in key or "column" in key):
        return "Stacked bar chart"
    if "bar" in key or "column" in key:
        return "Vertical bar chart"
    if "line" in key:
        return "Line chart"
    if "area" in key:
        return "Area chart"
    if "pie" in key or "donut" in key or "doughnut" in key:
        return "Pie chart"
    if "combo" in key or "combination" in key or "mixed" in key:
        return "Combo chart"
    return "Other"


def classify_image(image_path: Path, model: str, max_retries: int = 4) -> str:
    """Classify a single image, with retries on transient errors."""
    client = get_model_client(model)
    base64_image = encode_image_base64(image_path)
    image_type = get_image_type(image_path)
    data_url = f"data:image/{image_type};base64,{base64_image}"

    delay = 1.0
    last_err = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                            },
                        ],
                    }
                ],
                temperature=0,
            )
            content = response.choices[0].message.content or ""
            return normalize_label(content)
        except Exception as e:  # noqa: BLE001
            last_err = e
            logger.warning(
                f"[{image_path.name}] attempt {attempt + 1}/{max_retries} failed: {e}"
            )
            time.sleep(delay)
            delay *= 2
    logger.error(f"[{image_path.name}] giving up after {max_retries} retries: {last_err}")
    return "Other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--limit", type=int, default=None, help="Optional cap on number of images (debug)"
    )
    args = parser.parse_args()

    load_dotenv()
    if not os.environ.get("GEMINI_API_KEY") and args.model.startswith("gemini"):
        sys.exit("GEMINI_API_KEY not set in environment / .env")

    image_dir: Path = args.image_dir
    if not image_dir.exists():
        sys.exit(f"image_dir does not exist: {image_dir}")

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume from existing output (per-image caching).
    labels: Dict[str, str] = {}
    if output_path.exists():
        try:
            labels = json.loads(output_path.read_text())
            print(f"Resuming: loaded {len(labels)} cached labels from {output_path}")
        except json.JSONDecodeError:
            print(f"Could not parse existing {output_path}; starting fresh")

    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if args.limit:
        images = images[: args.limit]
    todo = [p for p in images if p.name not in labels]
    print(f"{len(images)} images total, {len(todo)} to classify with {args.model}")

    if not todo:
        print("Nothing to do.")
        return

    save_every = 25
    completed_since_save = 0

    def _work(p: Path):
        return p.name, classify_image(p, args.model)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_work, p): p for p in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Classifying"):
            name, label = fut.result()
            labels[name] = label
            completed_since_save += 1
            if completed_since_save >= save_every:
                output_path.write_text(json.dumps(labels, indent=2, sort_keys=True))
                completed_since_save = 0

    output_path.write_text(json.dumps(labels, indent=2, sort_keys=True))

    # Quick distribution print
    counts: Dict[str, int] = {}
    for v in labels.values():
        counts[v] = counts.get(v, 0) + 1
    print("\nLabel distribution:")
    for label in LABELS:
        print(f"  {label:25s}  {counts.get(label, 0)}")
    print(f"\nWrote {len(labels)} labels to {output_path}")


if __name__ == "__main__":
    main()
