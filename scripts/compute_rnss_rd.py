"""Compute RNSS and RD for a prediction run directory against a GT directory.

RNSS (Relative Number Set Similarity), following ChartOCR (Luo et al., WACV 2021):
each table is treated as a multiset of numeric values (headers/index excluded).
We build a relative-distance matrix between predicted and ground-truth values,
  dist(p, g) = min(1, |p - g| / |g|)   (with g = 0 handled explicitly),
find a minimum-cost bipartite matching (padding the smaller set with dummy
entries whose distance is 1), and report

  RNSS  = 1 - ( sum_{matched} dist + |N - M| ) / max(N, M)

so that both value errors and cardinality mismatches are penalised.

RD (Relative Deviation): the mean of dist(p, g) over the matched *real* value
pairs (lower is better) -- it isolates numeric closeness from cardinality.

Both are averaged over charts and reported in percent (RNSS x100, RD x100).

Usage:
    python scripts/compute_rnss_rd.py <run_dir> <gt_dir>
"""

import csv
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import optimize

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.eval_chart2table import _get_table_datapoints, _parse_table, _to_float
from src.utils import normalize_tsv_table


def _table_values(tsv_text: str):
    """Extract the multiset of numeric value cells from a TSV table."""
    table = _parse_table("title\t\n" + tsv_text.strip().lower(), transposed=False)
    values = []
    for _key, cell in _get_table_datapoints(table).items():
        if _key == "title":
            continue
        v = _to_float(cell)
        if v is not None:
            values.append(v)
    return values


def _rel_dist(p: float, g: float) -> float:
    if g == 0:
        return 0.0 if p == 0 else 1.0
    return min(1.0, abs(p - g) / abs(g))


def rnss_rd_for_pair(pred_values, gt_values):
    """Return (rnss, rd, n_matched) for one chart."""
    n, m = len(pred_values), len(gt_values)
    if n == 0 and m == 0:
        return 1.0, 0.0, 0
    if n == 0 or m == 0:
        return 0.0, float("nan"), 0

    # Real cost matrix of relative distances.
    cost = np.ones((n, m), dtype=float)
    for i, p in enumerate(pred_values):
        for j, g in enumerate(gt_values):
            cost[i, j] = _rel_dist(p, g)

    # Pad to a square (max x max) matrix; dummy pairs cost 1.0.
    size = max(n, m)
    padded = np.ones((size, size), dtype=float)
    padded[:n, :m] = cost
    row_ind, col_ind = optimize.linear_sum_assignment(padded)

    total = 0.0
    matched_real = []
    for r, c in zip(row_ind, col_ind):
        if r < n and c < m:
            d = cost[r, c]
            total += d
            matched_real.append(d)
        else:
            total += 1.0  # dummy / unmatched

    rnss = 1.0 - total / size
    rd = float(np.mean(matched_real)) if matched_real else float("nan")
    return rnss, rd, len(matched_real)


def _load_predictions(run_dir: Path):
    with (run_dir / "predictions.json").open(encoding="utf-8") as f:
        preds = json.load(f)
    return {p["image"].split(".")[0]: p["answer"] for p in preds}


def _load_gt(gt_dir: Path):
    tables = {}
    for csv_file in gt_dir.glob("*.csv"):
        with csv_file.open(encoding="utf-8") as f:
            reader = csv.reader(io.StringIO(f.read()))
            out = io.StringIO()
            writer = csv.writer(out, delimiter="\t", lineterminator="\n")
            for row in reader:
                writer.writerow(row)
            tables[csv_file.name.split(".")[0]] = normalize_tsv_table(out.getvalue())
    return tables


def compute(run_dir: Path, gt_dir: Path):
    preds = _load_predictions(run_dir)
    gts = _load_gt(gt_dir)

    rnss_scores, rd_scores = [], []
    for img, pred in preds.items():
        gt = gts.get(img)
        if gt is None:
            continue
        pred = re.sub(r"[ ]{2,}", "\t", normalize_tsv_table(pred))
        pv = _table_values(pred)
        gv = _table_values(gt)
        rnss, rd, _ = rnss_rd_for_pair(pv, gv)
        rnss_scores.append(rnss)
        if not np.isnan(rd):
            rd_scores.append(rd)

    return {
        "n": len(rnss_scores),
        "RNSS": 100.0 * float(np.mean(rnss_scores)) if rnss_scores else 0.0,
        "RD": 100.0 * float(np.mean(rd_scores)) if rd_scores else float("nan"),
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/compute_rnss_rd.py <run_dir> <gt_dir>")
        sys.exit(1)
    res = compute(Path(sys.argv[1]), Path(sys.argv[2]))
    print(json.dumps(res, indent=2))
