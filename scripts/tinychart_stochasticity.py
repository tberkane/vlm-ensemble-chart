"""Compute stochasticity stats across 19 TinyChart runs on WB v2:
  - For each chart, did at least one run differ from the others?
  - For each chart, do the 19 runs produce different table shapes (rows x cols)?

Used to back the intro claim about VLM run-to-run variability.
"""

import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))


def parse_shape(answer: str):
    """Return (n_rows, n_cols) of a TSV-like answer.
    Header counts as a row; row labels count as a col."""
    lines = [ln for ln in answer.strip().split("\n") if ln.strip()]
    if not lines:
        return (0, 0)
    n_rows = len(lines)
    # cells per line: number of tabs + 1
    n_cols = max(line.count("\t") + 1 for line in lines)
    return (n_rows, n_cols)


def main() -> None:
    manifest_path = project_root / "outputs" / "TinyChart" / "converted_run_dirs.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        run_dirs = [project_root / d for d in json.load(f)]

    # image -> list of 19 answers
    answers = {}
    for rd in run_dirs:
        with (rd / "predictions.json").open("r", encoding="utf-8") as f:
            preds = json.load(f)
        for p in preds:
            answers.setdefault(p["image"], []).append(p["answer"])

    n_charts = len(answers)
    n_runs = len(run_dirs)
    print(f"Charts: {n_charts}, runs/chart: {n_runs}")

    diff_count = 0
    shape_diff_count = 0
    for img, ans_list in answers.items():
        if len(set(ans_list)) > 1:
            diff_count += 1
        shapes = {parse_shape(a) for a in ans_list}
        if len(shapes) > 1:
            shape_diff_count += 1

    pct_diff = 100.0 * diff_count / n_charts
    pct_shape = 100.0 * shape_diff_count / n_charts
    print(f"Charts where >=1 run differs from the others: {diff_count}/{n_charts} "
          f"({pct_diff:.2f}%)")
    print(f"Charts with structural variation (different table shapes): "
          f"{shape_diff_count}/{n_charts} ({pct_shape:.2f}%)")

    # Also dump per-chart stats for further analysis
    out = {
        "n_charts": n_charts,
        "n_runs_per_chart": n_runs,
        "pct_charts_with_text_variation": pct_diff,
        "pct_charts_with_shape_variation": pct_shape,
        "n_charts_with_text_variation": diff_count,
        "n_charts_with_shape_variation": shape_diff_count,
    }
    out_path = project_root / "outputs" / "TinyChart" / "stochasticity_stats.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved stats: {out_path.relative_to(project_root)}")


if __name__ == "__main__":
    main()
