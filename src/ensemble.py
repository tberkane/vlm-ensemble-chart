import csv
import io
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils import normalize_tsv_table

logging.basicConfig(
    # filename="ensemble.log",
    level=logging.INFO,
    format="[%(levelname)s %(asctime)s %(module)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)


def ensemble_structured_tsvs(
    tsvs: List[str], structure: str, methods: List[str]
) -> Dict[str, str]:
    """Ensemble TSVs. Assume the TSVs all have the same row and column labels.
    Args:
        tsvs: List of TSVs to ensemble
        structure: Structure of the TSVs (TSV string with same row/column labels)
        methods: List of aggregation methods to use for ensemble
    Returns:
        Dictionary mapping method name to ensemble TSV string
    """
    logger.debug("Starting ensemble_structured_tsvs with %d input TSVs.", len(tsvs))
    if not tsvs:
        raise ValueError("No TSVs provided")

    # Parse structure to get canonical rows and columns
    structure_reader = csv.reader(io.StringIO(structure), delimiter="\t")
    structure_rows = list(structure_reader)

    # First row is header (column labels), first column is row labels
    structure_header = structure_rows[0]
    canon_cols = (
        structure_header[1:] if len(structure_header) > 1 else []
    )  # Column labels (skip first cell)
    canon_rows = [
        row[0] for row in structure_rows[1:] if len(row) > 0
    ]  # First column of each data row

    logger.debug(
        "Structure has %d rows and %d columns", len(canon_rows), len(canon_cols)
    )

    tsvs = [normalize_tsv_table(csv) for csv in tsvs]

    # Collect cell values from all TSVs
    # Key: (row_label, col_label) -> list of values
    cell_values: Dict[Tuple[str, str], List[float]] = {}

    for tsv_idx, tsv in enumerate(tsvs):
        tsv_reader = csv.reader(io.StringIO(tsv), delimiter="\t")
        tsv_rows = list(tsv_reader)

        # Extract values for each cell
        for row_idx, row in enumerate(tsv_rows[1:], start=1):  # Skip header row
            if len(row) == 0:
                continue
            row_label = row[0]

            for col_idx, cell_value in enumerate(
                row[1:], start=1
            ):  # Skip first column (row label)
                if col_idx - 1 >= len(canon_cols):
                    continue

                col_label = canon_cols[col_idx - 1]

                # Try to parse as float, skip if not numeric
                try:
                    if cell_value.strip():  # Only process non-empty cells
                        val = float(cell_value)
                        if pd.isna(val):
                            continue
                        key = (row_label, col_label)
                        cell_values.setdefault(key, []).append(val)
                except (ValueError, TypeError):
                    # Skip non-numeric values
                    continue

    logger.debug(
        "Collected values for %d canonical cells to aggregate.", len(cell_values)
    )

    aggregation_methods = list(set(methods + ["median", "mad"]))

    # Build output TSVs for each method
    # Use the exact header from structure to preserve format
    header = "\t".join(structure_header)
    tsv_lines = {method: [header] for method in aggregation_methods}

    for r in canon_rows:
        row_vals = {method: [] for method in aggregation_methods}
        for c in canon_cols:
            vals = cell_values.get((r, c), [])
            if not vals:
                for method in aggregation_methods:
                    row_vals[method].append("")
            else:
                vals_array = np.array(vals, dtype=float)
                med = float(np.median(vals_array))
                med = round(med, 6)
                row_vals["median"].append(str(med))

                # Compute MAD: median absolute deviation from median
                deviations = np.abs(vals_array - med)
                mad = float(np.median(deviations))
                mad = round(mad, 6)
                row_vals["mad"].append(str(mad))

                for method in aggregation_methods:
                    if method == "median" or method == "mad":
                        continue
                    if method == "mean":
                        agg_val = round(float(np.mean(vals_array)), 6)
                    elif method == "medoid":
                        # 1D medoid: element with minimal total absolute deviation
                        # is any element closest to the median -> O(n)
                        medoid_val = float(np.quantile(vals_array, 0.5, method="lower"))
                        agg_val = round(float(medoid_val), 6)
                    elif method == "huber":
                        if mad == 0 or np.allclose(vals_array, vals_array[0]):
                            huber_val = float(med)
                        else:
                            delta = 1.345 * mad
                            mu = float(med)

                            for _ in range(10):
                                rr = vals_array - mu
                                abs_r = np.abs(rr)

                                # Weights: 1 for inliers, delta/|r| for outliers
                                w = np.ones_like(vals_array, dtype=float)
                                mask = abs_r > delta
                                w[mask] = delta / abs_r[mask]

                                mu_new = float(np.sum(w * vals_array) / np.sum(w))

                                if abs(mu_new - mu) < 1e-6:
                                    break
                                mu = mu_new
                            huber_val = mu
                        agg_val = round(float(huber_val), 6)
                    elif method == "weighted_confidence":
                        # Retain top 60% most consistent extractions
                        # (closest to the median) and take their mean
                        deviations_from_med = np.abs(vals_array - med)
                        keep_n = max(1, int(len(vals_array) * 0.6))
                        keep_idx = np.argsort(deviations_from_med)[:keep_n]
                        agg_val = round(float(np.mean(vals_array[keep_idx])), 6)
                    elif method == "ransac":
                        # Remove outliers (>2 MAD from median) and take mean of inliers
                        if mad == 0 or np.allclose(vals_array, vals_array[0]):
                            agg_val = round(float(med), 6)
                        else:
                            inlier_mask = np.abs(vals_array - med) <= 2.0 * mad
                            if np.any(inlier_mask):
                                agg_val = round(float(np.mean(vals_array[inlier_mask])), 6)
                            else:
                                agg_val = round(float(med), 6)
                    else:
                        raise ValueError(f"Unsupported aggregation method: {method}")
                    row_vals[method].append(str(agg_val))

        for method in aggregation_methods:
            line = "\t".join(row_vals[method])
            tsv_lines[method].append(f"{r}\t{line}")

    return {method: "\n".join(lines) for method, lines in tsv_lines.items()}
