"""Compute error breakdown (correct/value/label/missing/extra) for a run dir.

Usage:
    python scripts/compute_error_breakdown.py <run_dir> <gt_dir>
"""

import csv
import io
import json
import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.eval_chart2table import chart2table_evaluator
from src.utils import normalize_tsv_table


def compute_breakdown(run_dir: Path, gt_dir: Path) -> dict:
    predictions_path = run_dir / "predictions.json"
    with predictions_path.open("r", encoding="utf-8") as f:
        predictions = json.load(f)
    predictions_dict = {
        prediction["image"].split(".")[0]: prediction["answer"]
        for prediction in predictions
    }

    tables_dict = {}
    for csv_file in gt_dir.glob("*.csv"):
        with csv_file.open("r", encoding="utf-8") as f:
            csv_content = f.read()
            reader = csv.reader(io.StringIO(csv_content))
            output = io.StringIO()
            writer = csv.writer(output, delimiter="\t", lineterminator="\n")
            for row in reader:
                writer.writerow(row)
            tsv_content = output.getvalue()
            tsv_content = normalize_tsv_table(tsv_content)
            tables_dict[csv_file.name.split(".")[0]] = tsv_content

    preds_and_gts = []
    for idx, (image, prediction) in enumerate(predictions_dict.items()):
        table = tables_dict.get(image)
        if table is None:
            continue
        prediction = normalize_tsv_table(prediction)
        prediction = re.sub(r"[ ]{2,}", "\t", prediction)
        preds_and_gts.append(
            {
                "model_answer": prediction,
                "gt_answer": table,
                "prediction_index": idx,
                "image": image,
            }
        )

    metrics = chart2table_evaluator(preds_and_gts, disable_tqdm=True)
    return metrics


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compute_error_breakdown.py <run_dir> <gt_dir>")
        sys.exit(1)
    run_dir = Path(sys.argv[1])
    gt_dir = Path(sys.argv[2])
    metrics = compute_breakdown(run_dir, gt_dir)
    print(json.dumps(metrics, indent=2))
