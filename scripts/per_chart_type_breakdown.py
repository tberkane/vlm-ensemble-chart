"""
Compute per-chart-type RMS-F1 breakdown for each model on ChartQA.

Reads:
  - chart-type labels JSON  (outputs/chartqa_chart_types.json)
  - prediction JSONs for each model
  - ground-truth tables

Outputs a CSV table and prints a LaTeX-ready table.

Usage:
    python scripts/per_chart_type_breakdown.py
"""

import csv
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.eval_chart2table import chart2table_evaluator
from src.utils import normalize_tsv_table

# ---------------------------------------------------------------------------
# Configuration: model name → prediction source
# ---------------------------------------------------------------------------
# Each entry: (display_name, path_to_predictions_json, is_adaptive_format)
MODELS = [
    (
        "TinyChart",
        project_root / "outputs/predict/2025-12-26_050703__ChartQA_Dataset_TinyChart_temp0_/predictions.json",
        False,
    ),
    (
        "OneChart",
        project_root / "outputs/predict/2025-12-26_112501__ChartQA_Dataset_OneChart_temp0/predictions.json",
        False,
    ),
    (
        "Qwen3-VL 235B",
        project_root / "outputs/adaptive/2026-02-19_135521__ChartQA_Dataset_test_50_percent__qwen_qwen3-vl-235b-a22b-instruct_temp1p0_maxsamp1_pat1_medoid/predictions.json",
        True,
    ),
    (
        "Llama 4 Maverick",
        project_root / "outputs/predict/2025-12-09_133154__ChartQA_Dataset_test_llama-4-maverick-17b-128e-instruct_temp0p0_CHART_EXTRACTION_PROMPT/predictions.json",
        False,
    ),
    (
        "GPT-5.1",
        project_root / "outputs/predict/2025-12-09_142125__ChartQA_Dataset_test_gpt-5.1_temp0p0_CHART_EXTRACTION_PROMPT/predictions.json",
        False,
    ),
]

CHART_TYPE_LABELS_PATH = project_root / "outputs/chartqa_chart_types.json"
GT_DIR = project_root / "data/ChartQA Dataset/test/tables"

# Order for the output table (skip types with 0 images)
TYPE_ORDER = [
    "Vertical bar chart",
    "Horizontal bar chart",
    "Stacked bar chart",
    "Line chart",
    "Pie chart",
    # Area chart / Combo chart / Other — only if they have images
]


def load_ground_truth(gt_dir: Path) -> dict:
    """Load ground-truth TSV tables keyed by stem (no extension)."""
    tables = {}
    for csv_file in gt_dir.glob("*.csv"):
        with csv_file.open("r", encoding="utf-8") as f:
            csv_content = f.read()
            reader = csv.reader(io.StringIO(csv_content))
            output = io.StringIO()
            writer = csv.writer(output, delimiter="\t", lineterminator="\n")
            for row in reader:
                writer.writerow(row)
            tsv_content = output.getvalue()
            tables[csv_file.stem] = normalize_tsv_table(tsv_content)
    return tables


def load_predictions(pred_path: Path) -> dict:
    """Load predictions keyed by image stem."""
    with pred_path.open("r", encoding="utf-8") as f:
        preds = json.load(f)
    return {
        p["image"].split(".")[0]: normalize_tsv_table(
            re.sub(r"[ ]{2,}", "\t", p["answer"])
        )
        for p in preds
        if p.get("answer", "").strip()
    }


def eval_subset(preds: dict, gt: dict, image_stems: list) -> float:
    """Compute RMS-F1 on a subset of images."""
    data = []
    for idx, stem in enumerate(image_stems):
        if stem in preds and stem in gt:
            data.append(
                {
                    "model_answer": preds[stem],
                    "gt_answer": gt[stem],
                    "prediction_index": idx,
                    "image": stem,
                }
            )
    if not data:
        return float("nan")
    metrics = chart2table_evaluator(data, disable_tqdm=True)
    return metrics["table_datapoints_f1"]


def main():
    # Load chart-type labels
    labels = json.loads(CHART_TYPE_LABELS_PATH.read_text())
    # key: image filename (with .png), value: chart type

    # Group image stems by chart type
    type_to_stems: dict[str, list[str]] = defaultdict(list)
    for filename, chart_type in labels.items():
        stem = filename.split(".")[0]
        type_to_stems[chart_type].append(stem)

    # Determine which types to report (non-empty, in order)
    active_types = [t for t in TYPE_ORDER if len(type_to_stems.get(t, [])) > 0]
    # Add any remaining types not in TYPE_ORDER
    for t in sorted(type_to_stems.keys()):
        if t not in active_types and len(type_to_stems[t]) > 0:
            active_types.append(t)

    all_stems = [stem for stems in type_to_stems.values() for stem in stems]

    print(f"Loaded {len(labels)} chart-type labels across {len(active_types)} types")
    for t in active_types:
        print(f"  {t:25s}  n={len(type_to_stems[t])}")

    # Load ground truth
    gt = load_ground_truth(GT_DIR)
    print(f"Loaded {len(gt)} ground-truth tables")

    # Results: model → type → f1
    results = {}

    for model_name, pred_path, _ in MODELS:
        print(f"\n--- {model_name} ---")
        if not pred_path.exists():
            print(f"  WARNING: predictions not found at {pred_path}")
            continue
        preds = load_predictions(pred_path)
        print(f"  Loaded {len(preds)} predictions")

        model_results = {}
        for chart_type in active_types:
            stems = type_to_stems[chart_type]
            # Filter to stems that exist in this model's predictions
            valid_stems = [s for s in stems if s in preds]
            f1 = eval_subset(preds, gt, valid_stems)
            model_results[chart_type] = f1
            print(f"  {chart_type:25s}  n={len(valid_stems):4d}  F1={f1:.2f}")

        # Overall (all images that have predictions)
        valid_all = [s for s in all_stems if s in preds]
        overall_f1 = eval_subset(preds, gt, valid_all)
        model_results["Overall"] = overall_f1
        print(f"  {'Overall':25s}  n={len(valid_all):4d}  F1={overall_f1:.2f}")

        results[model_name] = model_results

    # -----------------------------------------------------------------------
    # Print LaTeX table
    # -----------------------------------------------------------------------
    model_names = [m[0] for m in MODELS if m[0] in results]
    print("\n\n=== LaTeX Table ===\n")
    # Header
    cols = " & ".join([f"\\textbf{{{m}}}" for m in model_names])
    print(f"\\textbf{{Chart type}} & \\textbf{{n}} & {cols} \\\\")
    print("\\midrule")
    for chart_type in active_types:
        n = len(type_to_stems[chart_type])
        vals = []
        for m in model_names:
            f1 = results[m].get(chart_type, float("nan"))
            vals.append(f"{f1:.2f}" if f1 == f1 else "--")
        row = " & ".join(vals)
        print(f"{chart_type} & {n} & {row} \\\\")
    # Overall
    print("\\midrule")
    vals = []
    for m in model_names:
        f1 = results[m].get("Overall", float("nan"))
        vals.append(f"{f1:.2f}" if f1 == f1 else "--")
    row = " & ".join(vals)
    n_all = len(all_stems)
    print(f"Overall & {n_all} & {row} \\\\")

    # -----------------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------------
    csv_path = project_root / "outputs/chartqa_per_type_breakdown.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chart_type", "n"] + model_names)
        for chart_type in active_types:
            n = len(type_to_stems[chart_type])
            row = [chart_type, n]
            for m in model_names:
                row.append(f"{results[m].get(chart_type, float('nan')):.2f}")
            writer.writerow(row)
        row = ["Overall", len(all_stems)]
        for m in model_names:
            row.append(f"{results[m].get('Overall', float('nan')):.2f}")
        writer.writerow(row)
    print(f"\nSaved CSV to {csv_path}")


if __name__ == "__main__":
    main()
