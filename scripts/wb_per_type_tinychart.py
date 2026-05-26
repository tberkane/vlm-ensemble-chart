"""Add a TinyChart column to the WB v2 per-chart-type breakdown.

For consistency with the Table 1 single-pass number (mean over 19 runs),
we compute per-type F1 on each of the 19 TinyChart runs and average.
"""

import csv
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.eval_chart2table import chart2table_evaluator
from src.utils import normalize_tsv_table

GT_DIR = project_root / "data/World Bank v2/test/tables"
METADATA = project_root / "data/World Bank v2/test/metadata.json"
MANIFEST = project_root / "outputs/TinyChart/converted_run_dirs.json"


def load_gt() -> dict:
    tables = {}
    for csv_file in GT_DIR.glob("*.csv"):
        with csv_file.open("r", encoding="utf-8") as f:
            reader = csv.reader(io.StringIO(f.read()))
            buf = io.StringIO()
            writer = csv.writer(buf, delimiter="\t", lineterminator="\n")
            for row in reader:
                writer.writerow(row)
            tables[csv_file.stem] = normalize_tsv_table(buf.getvalue())
    return tables


def load_preds(pred_path: Path) -> dict:
    preds = json.load(open(pred_path))
    return {
        p["image"].split(".")[0]: normalize_tsv_table(
            re.sub(r"[ ]{2,}", "\t", p["answer"])
        )
        for p in preds
        if p.get("answer", "").strip()
    }


def eval_subset(preds: dict, gt: dict, stems: list) -> float:
    data = []
    for idx, s in enumerate(stems):
        if s in preds and s in gt:
            data.append({
                "model_answer": preds[s], "gt_answer": gt[s],
                "prediction_index": idx, "image": s,
            })
    if not data:
        return float("nan")
    return chart2table_evaluator(data, disable_tqdm=True)["table_datapoints_f1"]


def main():
    md = json.load(open(METADATA))
    type_to_stems = defaultdict(list)
    for fn, meta in md.items():
        type_to_stems[meta["chart_type"]].append(fn.split(".")[0])

    # Match the order used in Table 5 of the paper
    type_order = ["grouped_bar", "stacked_bar", "line", "area"]
    pretty = {"grouped_bar": "Grouped bar", "stacked_bar": "Stacked bar",
              "line": "Line", "area": "Area"}

    gt = load_gt()
    print(f"GT tables: {len(gt)}")
    for t in type_order:
        print(f"  {t:12s}  n={len(type_to_stems[t])}")

    run_dirs = [project_root / d for d in json.load(open(MANIFEST))]
    print(f"\nEvaluating {len(run_dirs)} TinyChart runs per chart type...")

    # type -> list of F1s (one per run)
    type_to_f1s = {t: [] for t in type_order}
    overall_f1s = []

    for i, rd in enumerate(run_dirs):
        preds = load_preds(rd / "predictions.json")
        for t in type_order:
            f1 = eval_subset(preds, gt, type_to_stems[t])
            type_to_f1s[t].append(f1)
        overall_f1 = eval_subset(preds, gt, [s for ss in type_to_stems.values() for s in ss])
        overall_f1s.append(overall_f1)
        print(f"  run{i:02d}  overall={overall_f1:.2f}  " +
              "  ".join(f"{t}={type_to_f1s[t][-1]:.2f}" for t in type_order))

    print("\n=== Mean across 19 runs ===")
    for t in type_order:
        vals = type_to_f1s[t]
        mean = sum(vals) / len(vals)
        print(f"  {pretty[t]:12s}  mean={mean:.2f}  std={(sum((v-mean)**2 for v in vals)/len(vals))**0.5:.3f}")
    overall_mean = sum(overall_f1s) / len(overall_f1s)
    print(f"  {'Overall':12s}  mean={overall_mean:.2f}")

    # Output snippet for paper
    print("\n=== Paper snippet (TinyChart column for Table 5) ===")
    for t in type_order:
        print(f"{pretty[t]}: {sum(type_to_f1s[t])/len(type_to_f1s[t]):.2f}")
    print(f"Overall: {overall_mean:.2f}")


if __name__ == "__main__":
    main()
