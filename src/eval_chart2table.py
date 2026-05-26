"""Metrics functions for Chart and Table related tasks."""

import dataclasses
import itertools
import logging
from typing import Any, Dict, Optional, Tuple

import editdistance
import numpy as np
from scipy import optimize
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Cache for parsed tables to avoid redundant parsing
_parse_table_cache: dict[tuple[str, bool], "Table"] = {}

# Cache for edit distance calculations (using string representation of theta for hashing)
_anls_cache: dict[tuple[str, str, str], float] = {}


def clear_caches():
    """Clear all caches. Useful for memory management or when starting a new evaluation."""
    global _parse_table_cache, _anls_cache
    _parse_table_cache.clear()
    _anls_cache.clear()


def anls_metric(target: str, prediction: str, theta: float = 0.5):
    """Calculates ANLS for DocVQA.

    There does not seem to be an official evaluation script.
    Public implementation on which this implementation is based:
    https://github.com/herobd/layoutlmv2/blob/main/eval_docvqa.py#L92

    Original paper (see Eq 1): https://arxiv.org/pdf/1907.00490.pdf

    Args:
      target: Target string.
      prediction: Predicted string.
      theta: Filter threshold set to 0.5 for DocVQA.

    Returns:
      ANLS score.
    """
    cache_key = (target, prediction, str(theta))
    if cache_key in _anls_cache:
        return _anls_cache[cache_key]

    if not target and not prediction:
        result = 1.0
        _anls_cache[cache_key] = result
        return result
    if target == prediction:
        result = 1.0
        _anls_cache[cache_key] = result
        return result

    edit_distance = editdistance.eval(target, prediction)
    max_len = max(len(target), len(prediction))
    if max_len == 0:
        result = 1.0
        _anls_cache[cache_key] = result
        return result

    normalized_ld = edit_distance / max_len
    score = 1 - normalized_ld if normalized_ld < theta else 0
    _anls_cache[cache_key] = score
    return score


def _to_float(text: str):
    try:
        if text.endswith("%"):
            return float(text.rstrip("%")) / 100.0
        return float(text)
    except ValueError:
        return None


def _get_relative_distance(target, prediction, theta=0.1):
    """Returns min(1, |target-prediction|/|target|). Then thresholds by theta."""
    if not target and not prediction:
        return 0.0
    if not target:
        return int(not prediction)
    distance = min(abs((target - prediction) / target), 1)
    return distance if distance < theta else 1


def _get_value_penalty(
    target_value: str,
    pred_value: str,
    *,
    text_theta: float,
    number_theta: float,
) -> float:
    """
    Returns a penalty in [0, 1], where 0 means perfect match and 1 means maximally wrong.
    For numeric values, uses _get_relative_distance (thresholded).
    For string values, uses (1 - ANLS).
    """
    pred_float = _to_float(pred_value)
    target_float = _to_float(target_value)

    if pred_float is not None and target_float is not None:
        return float(_get_relative_distance(target_float, pred_float, number_theta))

    if target_value == pred_value:
        return 0.0

    val_anls = anls_metric(target_value, pred_value, text_theta)
    return float(1.0 - val_anls)


@dataclasses.dataclass(frozen=True)
class Table:
    """Helper class for the content of a markdown/TSV table."""

    title: Optional[str] = None
    headers: tuple[str, Ellipsis] = dataclasses.field(default_factory=tuple)
    rows: tuple[tuple[str, Ellipsis], Ellipsis] = dataclasses.field(
        default_factory=tuple
    )


def _parse_table(text, transposed=False):
    """Builds a table from a TSV representation."""
    cache_key = (text, transposed)
    if cache_key in _parse_table_cache:
        return _parse_table_cache[cache_key]

    lines = text.lower().splitlines()
    if not lines:
        result = Table()
        _parse_table_cache[cache_key] = result
        return result

    if lines[0].startswith("title\t"):
        title = lines[0][len("title\t") :].strip()
        offset = 1
    else:
        title = None
        offset = 0

    if len(lines) < offset + 1:
        result = Table(title=title)
        _parse_table_cache[cache_key] = result
        return result

    rows = []
    for line in lines[offset:]:
        rows.append(tuple(v.strip() for v in line.split("\t")))

    if transposed:
        rows = [tuple(row) for row in itertools.zip_longest(*rows, fillvalue="")]

    table = Table(title=title, headers=rows[0], rows=tuple(rows[1:]))
    _parse_table_cache[cache_key] = table
    return table


def _get_table_datapoints(table: Table) -> Dict[str, str]:
    """Extracts a dict of datapoints from a table."""
    datapoints: Dict[str, str] = {}
    if table.title is not None:
        datapoints["title"] = table.title
    if not table.rows or len(table.headers) <= 1:
        return datapoints
    for row in table.rows:
        for header, cell in zip(table.headers[1:], row[1:]):
            datapoints[f"{row[0]} {header}"] = cell
    return datapoints


def _table_datapoints_precision_recall_f1_and_breakdown(
    target_table: Table,
    prediction_table: Table,
    text_theta=0.5,
    number_theta=0.1,
) -> Tuple[float, float, float, Dict[str, float]]:
    """
    Returns (precision, recall, f1, breakdown_percent).

    Breakdown components (percent) sum to 100.0 (up to floating error):
      - correct: same as per-example F1 * 100
      - value_errors: numeric (or string) value penalties, weighted by key match strength
      - label_errors: penalties from imperfect key ANLS (when key ANLS > 0 but < 1)
      - missing_datapoints: GT datapoints with no matched prediction (key ANLS == 0 or unassigned)
      - extra_datapoints: predicted datapoints with no matched GT (key ANLS == 0 or unassigned)
    """
    target_items = list(_get_table_datapoints(target_table).items())
    pred_items = list(_get_table_datapoints(prediction_table).items())

    ng = len(target_items)
    npred = len(pred_items)
    denom = ng + npred

    # Perfectly empty on both sides
    if denom == 0:
        return (
            1.0,
            1.0,
            1.0,
            {
                "correct": 100.0,
                "value_errors": 0.0,
                "label_errors": 0.0,
                "missing_datapoints": 0.0,
                "extra_datapoints": 0.0,
            },
        )

    # Degenerate cases
    if ng == 0:
        # everything is extra
        return (
            0.0,
            1.0,
            0.0,
            {
                "correct": 0.0,
                "value_errors": 0.0,
                "label_errors": 0.0,
                "missing_datapoints": 0.0,
                "extra_datapoints": 100.0,
            },
        )
    if npred == 0:
        # everything is missing
        return (
            1.0,
            0.0,
            0.0,
            {
                "correct": 0.0,
                "value_errors": 0.0,
                "label_errors": 0.0,
                "missing_datapoints": 100.0,
                "extra_datapoints": 0.0,
            },
        )

    # Build key-distance matrix: cost = 1 - ANLS(key)
    target_keys = [k for k, _ in target_items]
    pred_keys = [k for k, _ in pred_items]
    cost_matrix = np.zeros((ng, npred), dtype=float)
    for i, tk in enumerate(target_keys):
        for j, pk in enumerate(pred_keys):
            cost_matrix[i, j] = 1.0 - anls_metric(tk, pk, text_theta)

    row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix)

    assigned_t = set(row_ind.tolist())
    assigned_p = set(col_ind.tolist())

    # Unassigned due to rectangular matrix
    missing_count = ng - len(assigned_t)
    extra_count = npred - len(assigned_p)

    score = 0.0  # "soft TP score" used in precision/recall/F1

    # Error masses that will be normalized by denom to get percentages.
    # Note: matched-pair label/value penalties contribute *twice* (see notes in prompt).
    label_mass = 0.0
    value_mass = 0.0
    missing_mass = float(missing_count)
    extra_mass = float(extra_count)

    for r, c in zip(row_ind, col_ind):
        tk, tv = target_items[r]
        pk, pv = pred_items[c]

        key_score = float(anls_metric(tk, pk, text_theta))

        if key_score <= 0.0:
            # Treat as no match: counts as one missing GT + one extra prediction.
            missing_mass += 1.0
            extra_mass += 1.0
            continue

        value_penalty = _get_value_penalty(
            tv, pv, text_theta=text_theta, number_theta=number_theta
        )
        # datapoint score is key_score * value_similarity
        # where value_similarity = (1 - value_penalty)
        dp_score = key_score * (1.0 - float(value_penalty))
        score += dp_score

        # Decompose the "pair error mass" so everything sums cleanly:
        # 1 - dp_score = (1 - key_score) + key_score * value_penalty
        # Since denom counts both GT and pred sides, we add *2* times these.
        label_mass += 2.0 * (1.0 - key_score)
        value_mass += 2.0 * (key_score * float(value_penalty))

    precision = score / npred if npred > 0 else 1.0
    recall = score / ng if ng > 0 else 1.0
    f1 = 0.0 if score == 0 else (2.0 * score) / (ng + npred)

    # Convert masses to percentages (must sum to 100)
    correct_mass = 2.0 * score
    # Remaining mass implied by denom - correct_mass should match label+value+missing+extra (up to float error)
    # We compute "correct" directly from f1 to ensure it matches the reported f1 exactly.
    correct_pct = 100.0 * f1
    scale = 100.0 / float(denom)

    breakdown = {
        "correct": correct_pct,
        "value_errors": value_mass * scale,
        "label_errors": label_mass * scale,
        "missing_datapoints": missing_mass * scale,
        "extra_datapoints": extra_mass * scale,
    }

    # Small numerical cleanup to enforce sum=100.0 (optional but nice)
    s = sum(breakdown.values())
    if denom > 0 and abs(s - 100.0) > 1e-6:
        # Put the residual into the largest error bucket (not "correct") to avoid drifting "correct".
        residual = 100.0 - s
        candidates = [
            "value_errors",
            "label_errors",
            "missing_datapoints",
            "extra_datapoints",
        ]
        k = max(candidates, key=lambda x: breakdown[x])
        breakdown[k] += residual

    return precision, recall, f1, breakdown


def table_datapoints_precision_recall_per_point(
    targets,
    predictions,
    text_theta=0.5,
    number_theta=0.1,
):
    """Computes precision/recall/F1 per example, plus an error-type breakdown that sums to 100 per example."""
    assert len(targets) == len(predictions)

    per_point_scores = {
        "precision": [],
        "recall": [],
        "f1": [],
        # breakdown components (each per-example, percent)
        "correct": [],
        "value_errors": [],
        "label_errors": [],
        "missing_datapoints": [],
        "extra_datapoints": [],
    }

    for pred, target in zip(predictions, targets):
        all_metrics = []

        # Pre-parse prediction tables once
        pred_tables = [
            _parse_table(pred, transposed=transposed) for transposed in [True, False]
        ]
        # Pre-parse target tables once
        target_tables = [_parse_table(t) for t in target]

        for pred_table in pred_tables:
            all_metrics.extend(
                [
                    _table_datapoints_precision_recall_f1_and_breakdown(
                        target_table,
                        pred_table,
                        text_theta,
                        number_theta,
                    )
                    for target_table in target_tables
                ]
            )

        p, r, f, b = max(all_metrics, key=lambda x: x[2])  # pick best by F1
        per_point_scores["precision"].append(p)
        per_point_scores["recall"].append(r)
        per_point_scores["f1"].append(f)

        per_point_scores["correct"].append(b["correct"])
        per_point_scores["value_errors"].append(b["value_errors"])
        per_point_scores["label_errors"].append(b["label_errors"])
        per_point_scores["missing_datapoints"].append(b["missing_datapoints"])
        per_point_scores["extra_datapoints"].append(b["extra_datapoints"])

    return per_point_scores


def table_datapoints_precision_recall(
    targets,
    predictions,
    text_theta=0.5,
    number_theta=0.1,
):
    """Aggregated version returning overall precision/recall/F1 and an average breakdown (sums to 100)."""
    score_dict = table_datapoints_precision_recall_per_point(
        targets, predictions, text_theta, number_theta
    )

    n = len(targets)
    result = {
        "table_datapoints_precision": 100.0 * sum(score_dict["precision"]) / n,
        "table_datapoints_recall": 100.0 * sum(score_dict["recall"]) / n,
        "table_datapoints_f1": 100.0 * sum(score_dict["f1"]) / n,
        "error_breakdown": {
            "correct": sum(score_dict["correct"]) / n,
            "value_errors": sum(score_dict["value_errors"]) / n,
            "label_errors": sum(score_dict["label_errors"]) / n,
            "missing_datapoints": sum(score_dict["missing_datapoints"]) / n,
            "extra_datapoints": sum(score_dict["extra_datapoints"]) / n,
        },
    }

    # Enforce exact 100.0 sum for the aggregated breakdown (numerical polish)
    b = result["error_breakdown"]
    s = sum(b.values())
    if abs(s - 100.0) > 1e-6:
        residual = 100.0 - s
        candidates = [
            "value_errors",
            "label_errors",
            "missing_datapoints",
            "extra_datapoints",
        ]
        k = max(candidates, key=lambda x: b[x])
        b[k] += residual

    return result


def chart2table_evaluator(data, disable_tqdm=False):
    """
    Evaluates chart2table outputs.

    Returns:
      dict with:
        - table_datapoints_f1 (float, in percent)
        - error_breakdown (dict, percents summing to 100.0)
        - table_datapoints_precision / recall (float, in percent)
    """
    refs = []
    hyps = []

    for idx, item in enumerate(tqdm(data, desc="Evaluating", disable=disable_tqdm)):
        ref = "title\t\n" + item["gt_answer"].strip().lower()
        refs.append([ref])

        hyp = "title\t\n" + item["model_answer"].strip().lower()
        hyps.append(hyp)

        logger.debug(f"Item {idx}:")
        logger.debug(f"Ground Truth Table:\n{ref}")
        logger.debug(f"Predicted Table:\n{hyp}")
        logger.debug("-" * 80)

    metrics = table_datapoints_precision_recall(refs, hyps)

    # If you want a human-readable printout here (optional), uncomment:
    b = metrics["error_breakdown"]
    print(f'F1: {metrics["table_datapoints_f1"]:.2f}')
    print(
        "Breakdown (%): "
        + ", ".join(
            [
                f"correct={b['correct']:.2f}",
                f"value_errors={b['value_errors']:.2f}",
                f"label_errors={b['label_errors']:.2f}",
                f"missing_datapoints={b['missing_datapoints']:.2f}",
                f"extra_datapoints={b['extra_datapoints']:.2f}",
            ]
        )
    )

    return metrics
