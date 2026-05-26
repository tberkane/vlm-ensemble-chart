"""
Aggregation methods for ensemble results.
"""

import logging
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.eval_chart2table import Table, _parse_table, _to_float

from .base import AggregationMethod, ExtractionResult

# Import mentor's ensemble methods
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
from scripts.ensemble import ensemble_dataframes, tsvs_to_dfs
from src.ensemble import ensemble_structured_tsvs

logger = logging.getLogger(__name__)


def _aggregate_with_method(
    results: List[ExtractionResult],
    method: str,
    row_sim_threshold: float,
    col_sim_threshold: float,
    fallback_tsv: Optional[str] = None,
    pruning: bool = True,
    pruning_threshold: float = 0.5,
    structure: Optional[str] = None,
) -> Tuple[str, Optional[Dict[Tuple[int, int], float]], bool]:
    """
    Shared aggregation logic for methods that use ensemble_dataframes.

    Args:
        results: List of extraction results to aggregate
        method: Aggregation method name ("median" or "medoid")
        row_sim_threshold: Similarity threshold for row label matching (0-1)
        col_sim_threshold: Similarity threshold for column label matching (0-1)
        fallback_tsv: Optional TSV to return if fuzzy matching fails.
                     If None, falls back to first valid extraction.
                     Recommended: pass last successful ensemble.
        pruning: Whether to prune clusters with low support.
        pruning_threshold: Threshold for pruning clusters
        structure: Optional TSV skeleton structure with "_" placeholders. If provided, uses structured ensemble method.
    Returns:
        Tuple of (aggregated TSV string, MAD map, used_fallback flag, MAD TSV string)
        - aggregated TSV string
        - MAD map for convergence checking
        - used_fallback: True if fuzzy matching failed and fallback was used
        - MAD TSV string
        Returns ("", {}, False, "") if all extractions failed or no valid data.
    """
    if not results:
        logger.warning("Aggregation called with no results")
        return "", {}, False, ""

    # Extract TSV strings, filtering out None and empty strings
    tsv_list = [r.tsv_data for r in results if r.tsv_data and r.tsv_data.strip()]

    num_failed = len(results) - len(tsv_list)
    if num_failed > 0:
        logger.warning(
            f"Aggregation: {num_failed}/{len(results)} extractions failed or empty, "
            f"using {len(tsv_list)} valid results"
        )

    if not tsv_list:
        logger.error(
            f"Aggregation: All {len(results)} extractions failed - returning empty result"
        )
        return "", {}, False, ""

    if len(tsv_list) == 1:
        # Single result - no aggregation needed, but return empty variance_map
        logger.debug("Aggregation: Only 1 valid result, no variance calculation")
        return tsv_list[0], {}, False, ""  # Not using fallback, just single sample

    if structure:
        results_dict = ensemble_structured_tsvs(tsv_list, structure, methods=[method])
        aggregated_tsv = results_dict[method]
        mad_tsv = results_dict["mad"]
    else:
        # Convert to DataFrames using mentor's method
        # This handles TSV parsing, normalization, and transposition detection
        try:
            dfs = tsvs_to_dfs(tsv_list)

            # Debug: Log sample headers/labels before fuzzy matching
            logger.debug(
                f"Aggregation: Converting {len(tsv_list)} samples to DataFrames"
            )
            for idx, tsv in enumerate(tsv_list[:3]):  # Show first 3 samples
                first_line = tsv.split("\n")[0] if "\n" in tsv else tsv
                logger.debug(f"  Sample {idx+1} header: {first_line[:100]}")
        except Exception as e:
            logger.error(f"Failed to convert TSVs to DataFrames: {e}")
            # Return first valid TSV if conversion fails
            return tsv_list[0], {}, True, ""  # Used fallback due to parsing failure

        # Aggregate using mentor's fuzzy matching + MAD calculation
        # This function:
        # 1. Builds canonical labels using fuzzy string matching
        # 2. Computes value-context vectors for better alignment
        # 3. Aggregates cells by canonical (row, col) pairs
        # 4. Calculates median and MAD for each cell
        try:
            results_dict = ensemble_dataframes(
                dfs,
                methods=[method],  # Will automatically include "mad" as well
                row_sim_threshold=row_sim_threshold,
                col_sim_threshold=col_sim_threshold,
                pruning=pruning,
                pruning_threshold=pruning_threshold,
            )
            aggregated_tsv = "X\t" + results_dict[method]
            mad_tsv = "X\t" + results_dict["mad"]

            # Debug: Log successful ensemble stats
            num_rows = len(aggregated_tsv.strip().split("\n")) - 1  # Subtract header
            logger.debug(
                f"Aggregation: Ensemble produced {num_rows} data rows from {len(tsv_list)} samples"
            )

            # Log warning if ensemble produced only header (no data rows)
            if len(aggregated_tsv.strip().split("\n")) <= 1:
                # Choose best fallback: non-empty last ensemble > first extraction
                has_valid_fallback = (
                    fallback_tsv
                    and fallback_tsv.strip()
                    and len(fallback_tsv.strip().split("\n")) > 1
                )

                if has_valid_fallback:
                    fallback_result = fallback_tsv
                    fallback_choice = "last successful ensemble (non-empty)"
                    fallback_rows = len(fallback_tsv.strip().split("\n")) - 1
                    logger.debug(
                        f"Using last ensemble with {fallback_rows} rows as fallback"
                    )
                else:
                    fallback_result = tsv_list[0]
                    fallback_choice = "first valid extraction"
                    first_rows = len(tsv_list[0].strip().split("\n")) - 1
                    logger.debug(
                        f"Using first extraction with {first_rows} rows as fallback"
                    )

                # logger.warning(
                #     f"Aggregation: Fuzzy matching produced empty result (header only) from "
                #     f"{len(tsv_list)} valid extractions. This may indicate inconsistent "
                #     f"row/column labels that couldn't be aligned with current thresholds "
                #     f"(row_sim={row_sim_threshold}, col_sim={col_sim_threshold}). "
                #     f"Falling back to {fallback_choice}."
                # )

                # Debug: Show sample previews to diagnose alignment failure
                logger.debug("Sample previews (first 3):")
                for idx, tsv in enumerate(tsv_list[:3]):
                    preview = "\n".join(tsv.split("\n")[:3])  # First 3 lines
                    logger.debug(f"  Sample {idx+1}:\n{preview}")

                return (
                    fallback_result,
                    {},
                    True,
                    "",
                )  # Used fallback - fuzzy matching failed
        except Exception as e:
            # Choose best fallback: non-empty last ensemble > first extraction
            has_valid_fallback = (
                fallback_tsv
                and fallback_tsv.strip()
                and len(fallback_tsv.strip().split("\n")) > 1
            )

            if has_valid_fallback:
                fallback_result = fallback_tsv
                fallback_choice = "last successful ensemble (non-empty)"
            else:
                fallback_result = tsv_list[0]
                fallback_choice = "first valid extraction"

            logger.error(
                f"Failed to ensemble dataframes: {e}. Falling back to {fallback_choice}."
            )

            # Debug: Show what samples caused the failure
            logger.debug(
                f"Ensemble failed with {len(tsv_list)} samples. First sample preview:"
            )
            if tsv_list:
                preview = "\n".join(tsv_list[0].split("\n")[:3])
                logger.debug(f"{preview}")

            return fallback_result, {}, True, ""  # Used fallback - exception occurred

    # Parse MAD TSV to create variance_map for convergence checking
    variance_map: Dict[Tuple[int, int], float] = {}
    try:
        mad_table = _parse_table(mad_tsv, transposed=False)
        for row_idx, row in enumerate(mad_table.rows):
            for col_idx, cell in enumerate(row):
                mad_value = _to_float(cell)
                if mad_value is not None:
                    variance_map[(row_idx, col_idx)] = mad_value
                else:
                    # Non-numeric cell: use 0.0 (no variance)
                    variance_map[(row_idx, col_idx)] = 0.0
    except Exception as e:
        logger.warning(f"Could not parse MAD table for variance map: {e}")
        # Return empty variance_map if parsing fails
        variance_map = {}

    # Debug: Log variance map stats
    if variance_map:
        avg_mad = sum(variance_map.values()) / len(variance_map)
        max_mad = max(variance_map.values())
        logger.debug(
            f"Variance map: {len(variance_map)} cells, avg MAD={avg_mad:.4f}, max MAD={max_mad:.4f}"
        )
    else:
        logger.debug("Variance map: empty (no MAD calculated)")

    return (
        aggregated_tsv,
        variance_map,
        False,
        mad_tsv,
    )  # Successfully aggregated, no fallback used


class MedianAggregation(AggregationMethod):
    """
    Median aggregation using mentor's fuzzy matching method.

    Uses ensemble_dataframes() from scripts/ensemble.py with:
    - Fuzzy label matching for rows and columns (handles mismatches in same-chart extractions)
    - Value-context similarity for better alignment
    - Robust MAD calculation for convergence checking
    """

    def __init__(
        self,
        row_sim_threshold: float = 0.5,
        col_sim_threshold: float = 0.5,
        pruning: bool = True,
        pruning_threshold: float = 0.5,
    ):
        """
        Args:
            row_sim_threshold: Similarity threshold for row label matching (0-1)
            col_sim_threshold: Similarity threshold for column label matching (0-1)
            pruning: Whether to prune clusters with low support
            pruning_threshold: Threshold for pruning clusters
        """
        self.row_sim_threshold = row_sim_threshold
        self.col_sim_threshold = col_sim_threshold
        self.pruning = pruning
        self.pruning_threshold = pruning_threshold

    def aggregate(
        self,
        results: List[ExtractionResult],
        fallback_tsv: Optional[str] = None,
        structure: Optional[str] = None,
    ) -> Tuple[str, Optional[Dict[Tuple[int, int], float]], bool, str]:
        """
        Aggregate using mentor's fuzzy matching method.

        This is critical because even extractions from the same chart/model
        can have different row/column names and orders!

        Args:
            results: List of extraction results to aggregate
            fallback_tsv: Optional TSV to return if fuzzy matching fails.
                         If None, falls back to first valid extraction.
                         Recommended: pass last successful ensemble.

        Returns:
            Tuple of (aggregated TSV string, MAD map, used_fallback flag, MAD TSV string)
            - aggregated TSV string
            - MAD map for convergence checking
            - used_fallback: True if fuzzy matching failed and fallback was used
            - MAD TSV string
            Returns ("", {}, False, "") if all extractions failed or no valid data.
        """
        return _aggregate_with_method(
            results,
            method="median",
            row_sim_threshold=self.row_sim_threshold,
            col_sim_threshold=self.col_sim_threshold,
            fallback_tsv=fallback_tsv,
            pruning=self.pruning,
            pruning_threshold=self.pruning_threshold,
            structure=structure,
        )


class MedoidAggregation(AggregationMethod):
    """
    Medoid aggregation using mentor's fuzzy matching method.

    Uses ensemble_dataframes() from scripts/ensemble.py with:
    - Fuzzy label matching for rows and columns (handles mismatches in same-chart extractions)
    - Value-context similarity for better alignment
    - Robust MAD calculation for convergence checking
    - Medoid selection: chooses the actual value closest to the median
    """

    def __init__(
        self,
        row_sim_threshold: float = 0.5,
        col_sim_threshold: float = 0.5,
        pruning: bool = True,
        pruning_threshold: float = 0.5,
    ):
        """
        Args:
            row_sim_threshold: Similarity threshold for row label matching (0-1)
            col_sim_threshold: Similarity threshold for column label matching (0-1)
        """
        self.row_sim_threshold = row_sim_threshold
        self.col_sim_threshold = col_sim_threshold
        self.pruning = pruning
        self.pruning_threshold = pruning_threshold

    def aggregate(
        self,
        results: List[ExtractionResult],
        fallback_tsv: Optional[str] = None,
        structure: Optional[str] = None,
    ) -> Tuple[str, Optional[Dict[Tuple[int, int], float]], bool, str]:
        """
        Aggregate using mentor's fuzzy matching method with medoid selection.

        This is critical because even extractions from the same chart/model
        can have different row/column names and orders!

        Args:
            results: List of extraction results to aggregate
            fallback_tsv: Optional TSV to return if fuzzy matching fails.
                         If None, falls back to first valid extraction.
                         Recommended: pass last successful ensemble.

        Returns:
            Tuple of (aggregated TSV string, MAD map, used_fallback flag, MAD TSV string)
            - aggregated TSV string
            - MAD map for convergence checking
            - used_fallback: True if fuzzy matching failed and fallback was used
            - MAD TSV string
            Returns ("", {}, False, "") if all extractions failed or no valid data.
        """
        return _aggregate_with_method(
            results,
            method="medoid",
            row_sim_threshold=self.row_sim_threshold,
            col_sim_threshold=self.col_sim_threshold,
            fallback_tsv=fallback_tsv,
            pruning=self.pruning,
            pruning_threshold=self.pruning_threshold,
            structure=structure,
        )


class MeanAggregation(AggregationMethod):
    """
    Mean aggregation using mentor's fuzzy matching method.

    Uses ensemble_dataframes() from scripts/ensemble.py with:
    - Fuzzy label matching for rows and columns (handles mismatches in same-chart extractions)
    - Value-context similarity for better alignment
    - Robust MAD calculation for convergence checking
    - Mean aggregation: computes arithmetic mean of numeric values
    """

    def __init__(
        self,
        row_sim_threshold: float = 0.5,
        col_sim_threshold: float = 0.5,
        pruning: bool = True,
        pruning_threshold: float = 0.5,
    ):
        """
        Args:
            row_sim_threshold: Similarity threshold for row label matching (0-1)
            col_sim_threshold: Similarity threshold for column label matching (0-1)
        """
        self.row_sim_threshold = row_sim_threshold
        self.col_sim_threshold = col_sim_threshold
        self.pruning = pruning
        self.pruning_threshold = pruning_threshold

    def aggregate(
        self,
        results: List[ExtractionResult],
        fallback_tsv: Optional[str] = None,
        structure: Optional[str] = None,
    ) -> Tuple[str, Optional[Dict[Tuple[int, int], float]], bool, str]:
        """
        Aggregate using mentor's fuzzy matching method with mean aggregation.

        This is critical because even extractions from the same chart/model
        can have different row/column names and orders!

        Args:
            results: List of extraction results to aggregate
            fallback_tsv: Optional TSV to return if fuzzy matching fails.
                         If None, falls back to first valid extraction.
                         Recommended: pass last successful ensemble.

        Returns:
            Tuple of (aggregated TSV string, MAD map, used_fallback flag, MAD TSV string)
            - aggregated TSV string
            - MAD map for convergence checking
            - used_fallback: True if fuzzy matching failed and fallback was used
            - MAD TSV string
            Returns ("", {}, False, "") if all extractions failed or no valid data.
        """
        return _aggregate_with_method(
            results,
            method="mean",
            row_sim_threshold=self.row_sim_threshold,
            col_sim_threshold=self.col_sim_threshold,
            fallback_tsv=fallback_tsv,
            pruning=self.pruning,
            pruning_threshold=self.pruning_threshold,
            structure=structure,
        )


class HuberAggregation(AggregationMethod):
    """
    Huber aggregation using mentor's fuzzy matching method.

    Uses ensemble_dataframes() from scripts/ensemble.py with:
    - Fuzzy label matching for rows and columns (handles mismatches in same-chart extractions)
    - Value-context similarity for better alignment
    - Robust MAD calculation for convergence checking
    - Huber aggregation: robust to outliers using iterative weighted mean with Huber loss
    """

    def __init__(
        self,
        row_sim_threshold: float = 0.5,
        col_sim_threshold: float = 0.5,
        pruning: bool = True,
        pruning_threshold: float = 0.5,
    ):
        """
        Args:
            row_sim_threshold: Similarity threshold for row label matching (0-1)
            col_sim_threshold: Similarity threshold for column label matching (0-1)
        """
        self.row_sim_threshold = row_sim_threshold
        self.col_sim_threshold = col_sim_threshold
        self.pruning = pruning
        self.pruning_threshold = pruning_threshold

    def aggregate(
        self,
        results: List[ExtractionResult],
        fallback_tsv: Optional[str] = None,
        structure: Optional[str] = None,
    ) -> Tuple[str, Optional[Dict[Tuple[int, int], float]], bool, str]:
        """
        Aggregate using mentor's fuzzy matching method with Huber aggregation.

        This is critical because even extractions from the same chart/model
        can have different row/column names and orders!

        Huber aggregation is robust to outliers by using an iterative weighted mean
        where outliers (beyond delta = 1.345 * MAD) are downweighted.

        Args:
            results: List of extraction results to aggregate
            fallback_tsv: Optional TSV to return if fuzzy matching fails.
                         If None, falls back to first valid extraction.
                         Recommended: pass last successful ensemble.

        Returns:
            Tuple of (aggregated TSV string, MAD map, used_fallback flag, MAD TSV string)
            - aggregated TSV string
            - MAD map for convergence checking
            - used_fallback: True if fuzzy matching failed and fallback was used
            - MAD TSV string
            Returns ("", {}, False, "") if all extractions failed or no valid data.
        """
        return _aggregate_with_method(
            results,
            method="huber",
            row_sim_threshold=self.row_sim_threshold,
            col_sim_threshold=self.col_sim_threshold,
            fallback_tsv=fallback_tsv,
            pruning=self.pruning,
            pruning_threshold=self.pruning_threshold,
            structure=structure,
        )


class WeightedConfidenceAggregation(AggregationMethod):
    """Retains the top 60% most consistent extractions (closest to ensemble median) and averages them."""

    def __init__(self, row_sim_threshold=0.5, col_sim_threshold=0.5, pruning=True, pruning_threshold=0.5):
        self.row_sim_threshold = row_sim_threshold
        self.col_sim_threshold = col_sim_threshold
        self.pruning = pruning
        self.pruning_threshold = pruning_threshold

    def aggregate(self, results, fallback_tsv=None, structure=None):
        return _aggregate_with_method(
            results, method="weighted_confidence",
            row_sim_threshold=self.row_sim_threshold, col_sim_threshold=self.col_sim_threshold,
            fallback_tsv=fallback_tsv, pruning=self.pruning, pruning_threshold=self.pruning_threshold,
            structure=structure,
        )


class RANSACAggregation(AggregationMethod):
    """Removes outliers (>2 MAD from median) and averages inliers."""

    def __init__(self, row_sim_threshold=0.5, col_sim_threshold=0.5, pruning=True, pruning_threshold=0.5):
        self.row_sim_threshold = row_sim_threshold
        self.col_sim_threshold = col_sim_threshold
        self.pruning = pruning
        self.pruning_threshold = pruning_threshold

    def aggregate(self, results, fallback_tsv=None, structure=None):
        return _aggregate_with_method(
            results, method="ransac",
            row_sim_threshold=self.row_sim_threshold, col_sim_threshold=self.col_sim_threshold,
            fallback_tsv=fallback_tsv, pruning=self.pruning, pruning_threshold=self.pruning_threshold,
            structure=structure,
        )
