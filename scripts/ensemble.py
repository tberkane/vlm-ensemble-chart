import io
import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from io import StringIO
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import yaml
from jsonargparse import CLI

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.config import EnsembleConfig
from src.eval_chart2table import (
    Table,
    _get_table_datapoints,
    _parse_table,
    _to_float,
    anls_metric,
)
from src.utils import normalize_tsv_table

logging.basicConfig(
    # filename="ensemble.log",
    level=logging.INFO,
    format="[%(levelname)s %(asctime)s %(module)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)


def compute_forbidden_pairs(
    dfs: List[pd.DataFrame], use_index: bool = True
) -> Set[Tuple[str, str]]:
    """
    Compute label pairs that come from the same table, so they must not
    end up in the same canonical group.

    If use_index=True, operate on row labels (df.index),
    otherwise on column labels (df.columns).
    """
    forbidden: Set[Tuple[str, str]] = set()
    for df in dfs:
        labels = list(df.index if use_index else df.columns)
        # for every unordered pair of *distinct* labels from this table
        for a, b in combinations(labels, 2):
            if a == b:
                continue
            # Use string representation to establish a deterministic but type-agnostic order
            a_str, b_str = str(a), str(b)
            key = (a, b) if a_str <= b_str else (b, a)
            forbidden.add(key)
    logger.debug(
        "Computed %d forbidden pairs from %d dataframes (use_index=%s)",
        len(forbidden),
        len(dfs),
        use_index,
    )
    return forbidden


def build_canonical_labels(
    labels: Iterable[str],
    sim_threshold: float = 0.5,
    forbidden_pairs: Optional[Set[Tuple[str, str]]] = None,
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    label_list = list(labels)
    logger.debug(
        "Building canonical labels (sim_threshold=%.2f), %d labels, forbidden_pairs=%s",
        sim_threshold,
        len(label_list),
        forbidden_pairs is not None,
    )

    # Count and deduplicate
    label_counts = Counter(label_list)
    unique_labels = list(dict.fromkeys(label_list))

    # Pre-normalize strings
    norm_label = {lab: str(lab).strip().lower() for lab in unique_labels}

    # Precompute forbidden adjacency sets
    forbidden_map: Dict[Any, Set[Any]] = {}
    if forbidden_pairs is not None:
        for a, b in forbidden_pairs:
            forbidden_map.setdefault(a, set()).add(b)
            forbidden_map.setdefault(b, set()).add(a)

    # Caches
    str_sim_cache: Dict[Tuple[str, str], float] = {}

    def get_str_sim(a_lab, b_lab) -> float:
        a_norm = norm_label[a_lab]
        b_norm = norm_label[b_lab]
        if not a_norm and not b_norm:
            return 1.0
        if not a_norm or not b_norm:
            return 0.0
        key = (a_norm, b_norm) if a_norm <= b_norm else (b_norm, a_norm)
        sim = str_sim_cache.get(key)
        if sim is not None:
            return sim
        sim = float(anls_metric(a_norm, b_norm))
        str_sim_cache[key] = sim
        return sim

    canonical_labels: List[str] = []
    temp_groups: Dict[str, List[str]] = {}
    temp_group_sets: Dict[str, Set[str]] = {}
    temp_label_to_canonical: Dict[str, str] = {}

    # Greedy clustering
    for lab in unique_labels:
        best_canon = None
        best_sim = 0.0

        lab_forbidden = forbidden_map.get(lab)

        for canon in canonical_labels:
            # forbidden check via set intersection
            if lab_forbidden is not None and temp_group_sets[canon] & lab_forbidden:
                continue

            sim = get_str_sim(lab, canon)

            if sim > best_sim:
                best_sim = sim
                best_canon = canon

        if best_canon is None or best_sim < sim_threshold:
            canonical_labels.append(lab)
            temp_groups[lab] = [lab]
            temp_group_sets[lab] = {lab}
            temp_label_to_canonical[lab] = lab
        else:
            temp_groups[best_canon].append(lab)
            temp_group_sets[best_canon].add(lab)
            temp_label_to_canonical[lab] = best_canon

    # Second pass: choose final canonical labels
    canonical_groups: Dict[str, List[str]] = {}
    label_to_canonical: Dict[str, str] = {}

    for temp_canon, members in temp_groups.items():
        best_label = max(members, key=lambda l: (label_counts[l], -members.index(l)))
        canonical_groups[best_label] = members
        for m in members:
            label_to_canonical[m] = best_label
        logger.debug("Canonical group '%s': members=%s", best_label, members)

    logger.debug("Total canonical groups formed: %d", len(canonical_groups))
    return label_to_canonical, canonical_groups


def ensemble_dataframes(
    dfs: List[pd.DataFrame],
    methods: List[str],
    row_sim_threshold: float = 0.5,
    col_sim_threshold: float = 0.5,
    pruning: bool = True,
    pruning_threshold: float = 0.5,
) -> Dict[str, str]:
    """
    Ensemble multiple TSV-like DataFrames into a consensus 'median' table.
    Assumes:
      - df.index contains row labels
      - df.columns contains column labels
      - cells are numeric

    Steps:
      1. Build canonical row and column labels across all DataFrames using
         greedy clustering on string similarity.
      2. Map each DataFrame's rows/cols to canonical labels.
      3. For each canonical (row, col), collect all numeric values across DataFrames.
      4. The consensus cell is the median of those values.
    """
    logger.debug("Starting ensemble_dataframes with %d input DataFrames.", len(dfs))
    if not dfs:
        raise ValueError("No DataFrames provided")

    # Collect all row/column labels across DataFrames
    all_row_labels: List[str] = []
    all_col_labels: List[str] = []
    for df in dfs:
        all_row_labels.extend(list(df.index))
        all_col_labels.extend(list(df.columns))

    logger.debug(
        "Discovered %d row labels and %d column labels in all DataFrames.",
        len(all_row_labels),
        len(all_col_labels),
    )

    # Compute forbidden pairs: labels that co-occur in the same table
    row_forbidden_pairs = compute_forbidden_pairs(dfs, use_index=True)
    col_forbidden_pairs = compute_forbidden_pairs(dfs, use_index=False)
    logger.debug(
        "Computed forbidden pairs (rows: %d, columns: %d)",
        len(row_forbidden_pairs),
        len(col_forbidden_pairs),
    )

    row_label_to_canon, row_canon_groups = build_canonical_labels(
        all_row_labels,
        sim_threshold=row_sim_threshold,
        forbidden_pairs=row_forbidden_pairs,
    )
    col_label_to_canon, col_canon_groups = build_canonical_labels(
        all_col_labels,
        sim_threshold=col_sim_threshold,
        forbidden_pairs=col_forbidden_pairs,
    )

    # --- Drop clusters whose support < pruning_threshold of the number of input CSVs ---

    num_tables = len(dfs)
    min_support = max(
        1, int(num_tables * pruning_threshold + 0.5)
    )  # must appear in >= pruning_threshold of the tables

    logger.debug(
        "Pruning clusters with < %d support tables (num_tables=%d)",
        min_support,
        num_tables,
    )

    # Row cluster support: in how many tables does each canonical row appear?
    row_support_tables: Dict[str, set] = {}
    for t_idx, df in enumerate(dfs):
        labels_in_df = set(df.index)
        for lab in labels_in_df:
            canon = row_label_to_canon.get(lab)
            if canon is None:
                continue
            row_support_tables.setdefault(canon, set()).add(t_idx)

    kept_row_canons = {
        canon
        for canon, tables in row_support_tables.items()
        if (len(tables) >= min_support) or not pruning
    }

    # Column cluster support: in how many tables does each canonical column appear?
    col_support_tables: Dict[str, set] = {}
    for t_idx, df in enumerate(dfs):
        labels_in_df = set(df.columns)
        for lab in labels_in_df:
            canon = col_label_to_canon.get(lab)
            if canon is None:
                continue
            col_support_tables.setdefault(canon, set()).add(t_idx)

    kept_col_canons = {
        canon
        for canon, tables in col_support_tables.items()
        if (len(tables) >= min_support) or not pruning
    }

    logger.debug(
        "Canonical row clusters kept: %d, canonical column clusters kept: %d",
        len(kept_row_canons),
        len(kept_col_canons),
    )

    # Prune mappings: labels whose canonical group has low support get dropped
    row_label_to_canon = {
        lab: canon
        for lab, canon in row_label_to_canon.items()
        if canon in kept_row_canons
    }
    col_label_to_canon = {
        lab: canon
        for lab, canon in col_label_to_canon.items()
        if canon in kept_col_canons
    }

    logger.debug(
        "After pruning, %d row, %d column canonical mappings remain.",
        len(row_label_to_canon),
        len(col_label_to_canon),
    )

    # Aggregation structure: (canon_row, canon_col) -> list of values
    cell_values: Dict[Tuple[str, str], List[float]] = {}

    # Collect numeric values across all tables without using df.at
    for df in dfs:
        # Materialize labels and values once
        rows = list(df.index)
        cols = list(df.columns)
        values = df.to_numpy()  # shape: (n_rows, n_cols)

        # Precompute canonical mapping for each row/column position
        row_canon = [row_label_to_canon.get(lbl) for lbl in rows]
        col_canon = [col_label_to_canon.get(lbl) for lbl in cols]

        # Iterate by integer position, read from NumPy array
        for i, canon_row in enumerate(row_canon):
            if canon_row is None:
                continue  # row belongs to a dropped/unsupported cluster

            row_vals = values[i, :]  # 1D view over columns

            for j, canon_col in enumerate(col_canon):
                if canon_col is None:
                    continue  # column belongs to a dropped/unsupported cluster

                val = row_vals[j]

                # NaNs are skipped
                if pd.isna(val):
                    continue

                key = (canon_row, canon_col)
                cell_values.setdefault(key, []).append(float(val))

    logger.debug(
        "Collected values for %d canonical cells to aggregate.", len(cell_values)
    )

    # Build consensus index/columns from canonical labels that actually appear
    canon_rows = sorted(
        {r for (r, c) in cell_values.keys()}, key=lambda x: str(x).strip().lower()
    )
    canon_cols = sorted(
        {c for (r, c) in cell_values.keys()}, key=lambda x: str(x).strip().lower()
    )

    aggregation_methods = list(set(methods + ["median", "mad"]))

    header = "\t".join(str(c) for c in canon_cols)
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
                        med = float(np.quantile(vals_array, 0.5, method="lower"))
                        agg_val = round(float(med), 6)
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
                    else:
                        raise ValueError(f"Unsupported aggregation method: {method}")
                    row_vals[method].append(str(agg_val))

        for method in aggregation_methods:
            line = "\t".join(row_vals[method])
            tsv_lines[method].append(f"{r}\t{line}")

    return {method: "\n".join(lines) for method, lines in tsv_lines.items()}


def repair_tsv_string(tsv_string: str) -> str:
    """Try to fix broken TSV rows by padding or truncating to the mean length."""
    lines = [line for line in tsv_string.strip().split("\n") if line.strip()]
    if not lines:
        logger.error("No lines could be parsed from TSV string.")
        raise ValueError("No lines could be parsed from TSV string.")
    # Split by tab
    row_lens = [len(row.split("\t")) for row in lines]
    if not row_lens:
        logger.error("No row lengths calculated for TSV fix-up.")
        raise ValueError("No row lengths calculated for TSV fix-up.")
    mean_len = int(round(sum(row_lens) / len(row_lens)))
    fixed_lines = []
    for row in lines:
        cols = row.split("\t")
        # Pad with empty strings
        if len(cols) < mean_len:
            logger.debug("Padding row from %d to %d columns", len(cols), mean_len)
            cols += [""] * (mean_len - len(cols))
        # Truncate if too long
        elif len(cols) > mean_len:
            logger.debug("Truncating row from %d to %d columns", len(cols), mean_len)
            cols = cols[:mean_len]
        fixed_lines.append("\t".join(cols))
    fixed_tsv_string = "\n".join(fixed_lines)
    return fixed_tsv_string


def repair_df_index(df: pd.DataFrame) -> pd.DataFrame:
    """Try to fix the index of a DataFrame by replacing NaNs with 0 or empty strings."""
    if df.index.hasnans:
        logger.debug("DataFrame has NaNs in the index; attempting to fix index.")
        # Check if at least one other element is a number
        index_wo_nan = [x for x in df.index if not pd.isna(x)]
        has_number = any(
            isinstance(y, (int, float, np.integer, np.floating))
            or (
                isinstance(y, str)
                and y.strip().replace(".", "", 1).replace("-", "", 1).isdigit()
            )
            for y in index_wo_nan
        )
        if has_number:
            logger.debug("Setting NaN index values to 0.")
            df.index = df.index.map(lambda x: 0 if pd.isna(x) else x)
        else:
            logger.debug("Setting NaN index values to empty string.")
            df.index = df.index.map(lambda x: "" if pd.isna(x) else x)
    return df


def dataframe_from_tsv_string(tsv_string: str, index_col: int = 0) -> pd.DataFrame:
    """
    Load a TSV string into a pandas DataFrame.
    """
    try:
        df = pd.read_csv(io.StringIO(tsv_string), index_col=index_col, delimiter="\t")
    except Exception as e:
        logger.debug("TSV reading failed: %s. Attempting to fix broken rows...", e)
        fixed_tsv_string = repair_tsv_string(tsv_string)

        df = pd.read_csv(
            io.StringIO(fixed_tsv_string), index_col=index_col, delimiter="\t"
        )
        logger.debug("Recovered DataFrame after TSV fix-up; shape: %s", df.shape)

    df = repair_df_index(df)
    return df


def table_to_tsv(table) -> str:
    """Convert a Table object to a TSV string."""
    lines = [list(table.headers)] + list(table.rows)
    result_lines = []
    for row in lines:
        row_line = "\t".join(row)
        result_lines.append(row_line)
    return "\n".join(result_lines)


def aggregate_answers(
    df_list: List[pd.DataFrame],
    methods: List[str],
    row_similarity_threshold: float = 0.5,
    column_similarity_threshold: float = 0.5,
) -> Dict[str, str]:
    """
    Aggregate TSV tables using ensemble method.

    Args:
        df_list: List of DataFrames to aggregate

    Returns:
        Dict of (aggregation method: TSV string)
    """
    methods = [method.lower() for method in methods]
    for method in methods:
        if method not in ["median", "mean", "medoid", "huber"]:
            raise ValueError(f"Unsupported aggregation method: {method}")

    logger.debug("Aggregating %d DataFrames", len(df_list))

    results = ensemble_dataframes(
        df_list,
        methods,
        row_sim_threshold=row_similarity_threshold,
        col_sim_threshold=column_similarity_threshold,
    )
    for result_k, result_v in results.items():
        results[result_k] = "X\t" + result_v
    return results


def tsvs_to_dfs(tsv_list: List[str]) -> List[pd.DataFrame]:
    normalized_tsv_list = [normalize_tsv_table(csv) for csv in tsv_list]

    parsed_tsv_list = []

    for normalized_tsv in normalized_tsv_list:
        table_normal = _parse_table(normalized_tsv, transposed=False)
        table_transposed = _parse_table(normalized_tsv, transposed=True)
        # Prefer the one with more rows
        if len(table_transposed.rows) > len(table_normal.rows):
            transposed_tsv = table_to_tsv(table_transposed)
            parsed_tsv_list.append(transposed_tsv)
        else:
            parsed_tsv_list.append(normalized_tsv)

    dfs = [dataframe_from_tsv_string(tsv_string) for tsv_string in parsed_tsv_list]
    return dfs


def main(cfg: EnsembleConfig) -> None:

    # 1. Normalize member_run_dirs
    member_dirs: List[Path] = []
    for d in cfg.member_run_dirs:
        p = Path(d)
        if not p.is_absolute():
            p = project_root / p
        if not p.exists():
            raise FileNotFoundError(f"Member run dir does not exist: {p}")
        member_dirs.append(p)

    logger.info(f"Ensembling over {len(member_dirs)} runs:")
    for p in member_dirs:
        logger.info(f"  - {p}")

    # 2. Load model information from each member run's config.yaml
    member_models: Dict[str, List[str]] = {}
    for original_dir_str, run_dir in zip(cfg.member_run_dirs, member_dirs):
        config_path = run_dir / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"config.yaml not found in {run_dir}")

        with config_path.open("r", encoding="utf-8") as f:
            run_config = yaml.safe_load(f)

        model = run_config.get("model")
        if model is None:
            raise ValueError(f"Model not found in config.yaml: {config_path}")

        # Use original member_run_dirs string as key to match the config format
        member_models[original_dir_str] = [model]

    # 3. Load predictions from each run
    #    Build: image_name -> [answer1, answer2, ...]
    answers_by_image: Dict[str, List[str]] = defaultdict(list)
    tokens_by_image: Dict[str, List[Dict[str, int]]] = defaultdict(list)

    for run_dir in member_dirs:
        pred_path = run_dir / "predictions.json"
        if not pred_path.exists():
            raise FileNotFoundError(f"predictions.json not found in {run_dir}")

        with pred_path.open("r", encoding="utf-8") as f:
            preds = json.load(f)

        for entry in preds:
            img = entry["image"]
            ans = entry["answer"]
            answers_by_image[img].append(ans)

            tokens_by_image[img].append(
                {
                    "input_tokens": entry["input_tokens"],
                    "output_tokens": entry["output_tokens"],
                }
            )

    logger.info(f"Found predictions for {len(answers_by_image)} distinct images.")

    # 4. Create ensemble run directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    tag = cfg.output_tag or "ensemble"
    run_name = f"{timestamp}__{tag}"
    ensemble_dir = project_root / "outputs" / "ensemble" / run_name
    ensemble_dir.mkdir(parents=True, exist_ok=False)
    logger.info(f"Ensemble run directory: {ensemble_dir}")

    # 5. Compute ensembled predictions
    ensembled_predictions: List[Dict[str, Any]] = []
    for i, (image_name, tsv_list) in enumerate(answers_by_image.items()):
        dfs = tsvs_to_dfs(tsv_list)
        ensembled = aggregate_answers(
            dfs,
            cfg.aggregation_methods,
            cfg.row_similarity_threshold,
            cfg.column_similarity_threshold,
        )

        # Store sum of tokens used across all members for this image
        token_dicts = tokens_by_image[image_name]
        input_tokens_sum = sum(t["input_tokens"] for t in token_dicts)
        output_tokens_sum = sum(t["output_tokens"] for t in token_dicts)

        result = {
            "image": image_name,
            "input_tokens": input_tokens_sum,
            "output_tokens": output_tokens_sum,
        }

        for method in cfg.aggregation_methods + ["mad"]:
            result[method] = ensembled[method]
        result["answer"] = ensembled["median"]
        ensembled_predictions.append(result)

    ensembled_predictions.sort(key=lambda x: x["image"])

    # 6. Save predictions.json
    pred_out_path = ensemble_dir / "predictions.json"
    with pred_out_path.open("w", encoding="utf-8") as f:
        json.dump(ensembled_predictions, f, indent=4)
    logger.info(f"Saved ensembled predictions to {pred_out_path}")

    # 7. Save ensemble config + source runs metadata
    # Build config dict with member_models
    cfg_dict = vars(cfg).copy()
    cfg_dict["member_models"] = member_models

    cfg_path = ensemble_dir / "config.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f, default_flow_style=False, sort_keys=False)

    src_runs_path = ensemble_dir / "source_runs.json"
    with src_runs_path.open("w", encoding="utf-8") as f:
        json.dump([str(d) for d in member_dirs], f, indent=4)
    logger.info(f"Saved ensemble metadata to {cfg_path} and {src_runs_path}")


if __name__ == "__main__":
    CLI(main)
