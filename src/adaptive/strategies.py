"""
Sampling strategies for adaptive ensemble extraction.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..extract_data import (
    conforms_to_structure,
    extract_data_from_chart,
    extract_structure_from_chart,
)
from .base import (
    AggregationMethod,
    ConvergenceChecker,
    EnsembleResult,
    ExtractionResult,
    IterationSnapshot,
    SamplingStrategy,
)

logger = logging.getLogger(__name__)


class FixedSamplingStrategy(SamplingStrategy):
    """Fixed number of samples (baseline)."""

    def __init__(
        self,
        num_samples: int,
        model: str,
        temperature: Optional[float] = None,
        aggregation: Optional[AggregationMethod] = None,
    ):
        """
        Args:
            num_samples: Number of samples to take
            model: Model to use for extraction
            temperature: Optional temperature override
            aggregation: Aggregation method (default: MedianAggregation)
        """
        self.num_samples = num_samples
        self.model = model
        self.temperature = temperature

        if aggregation is None:
            from .aggregation import MedianAggregation

            aggregation = MedianAggregation()
        self.aggregation = aggregation

    def extract(
        self,
        image_path: Path,
        convergence_checker: Optional[ConvergenceChecker] = None,
    ) -> EnsembleResult:
        """Extract with fixed number of samples."""
        results: List[ExtractionResult] = []
        iteration_history: List[IterationSnapshot] = []

        for i in range(self.num_samples):
            extraction = extract_data_from_chart(
                image_path,
                model=self.model,
                temperature=self.temperature,
            )

            result = ExtractionResult(
                tsv_data=extraction["csv_data"],
                input_tokens=extraction["input_tokens"],
                output_tokens=extraction["output_tokens"],
                model=self.model,
                metadata={"sample_idx": i},
            )
            results.append(result)

            # Log extraction failures
            if extraction["csv_data"] is None:
                logger.warning(
                    f"Fixed strategy: Sample {i+1}/{self.num_samples} failed for {image_path.name}"
                )

            # Aggregate current results to track progression
            iter_tsv, _, _ = self.aggregation.aggregate(results)
            iteration_history.append(
                IterationSnapshot(
                    iteration=len(results),
                    tsv_data=iter_tsv if iter_tsv else "",
                    converged=len(results) == self.num_samples,
                )
            )

        # Final aggregation
        aggregated_tsv, variance_map, _ = self.aggregation.aggregate(results)

        total_input_tokens = sum(r.input_tokens for r in results)
        total_output_tokens = sum(r.output_tokens for r in results)

        return EnsembleResult(
            tsv_data=aggregated_tsv,
            num_samples=len(results),
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            converged=False,  # Fixed strategy doesn't check convergence
            variance_map=variance_map,
            iteration_history=iteration_history,
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "strategy": "fixed",
            "num_samples": self.num_samples,
            "model": self.model,
            "temperature": self.temperature,
        }


class AdaptiveSamplingStrategy(SamplingStrategy):
    """Adaptive sampling with early stopping."""

    def __init__(
        self,
        min_samples: int,
        max_samples: int,
        model: str,
        convergence_checker: ConvergenceChecker,
        temperature: Optional[float] = None,
        aggregation: Optional[AggregationMethod] = None,
    ):
        """
        Args:
            min_samples: Minimum number of samples
            max_samples: Maximum number of samples
            model: Model to use for extraction
            convergence_checker: Convergence checker for early stopping
            temperature: Optional temperature override
            aggregation: Aggregation method (default: MedianAggregation)
        """
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.model = model
        self.convergence_checker = convergence_checker
        self.temperature = temperature

        if aggregation is None:
            from .aggregation import MedianAggregation

            aggregation = MedianAggregation()
        self.aggregation = aggregation

    def extract(
        self,
        image_path: Path,
        convergence_checker: Optional[ConvergenceChecker] = None,
    ) -> EnsembleResult:
        """Extract with adaptive early stopping."""
        # Use provided convergence_checker or default to instance one
        checker = convergence_checker or self.convergence_checker

        results: List[ExtractionResult] = []
        converged = False
        convergence_iteration = None
        iteration_history: List[IterationSnapshot] = []

        for i in range(self.max_samples):
            # Extract
            extraction = extract_data_from_chart(
                image_path,
                model=self.model,
                temperature=self.temperature,
            )

            result = ExtractionResult(
                tsv_data=extraction["csv_data"],
                input_tokens=extraction["input_tokens"],
                output_tokens=extraction["output_tokens"],
                model=self.model,
                metadata={"sample_idx": i},
            )
            results.append(result)

            # Log extraction failures
            if extraction["csv_data"] is None:
                logger.warning(
                    f"Adaptive strategy: Sample {i+1}/{self.max_samples} failed for {image_path.name}"
                )

            # Aggregate current results and check convergence
            aggregated_tsv, variance_map, _ = self.aggregation.aggregate(results)

            # Check convergence after min_samples
            has_converged_now = False
            if i >= self.min_samples - 1:
                # Only declare convergence if:
                # 1. Checker says raw results converged
                # 2. Aggregation produced non-empty result
                if checker.has_converged(results, variance_map):
                    if aggregated_tsv and aggregated_tsv.strip():
                        converged = True
                        has_converged_now = True
                        convergence_iteration = i + 1
                        logger.debug(
                            f"Converged at iteration {convergence_iteration} "
                            f"for {image_path.name}"
                        )
                    else:
                        logger.warning(
                            f"Convergence criteria met but aggregation returned empty result "
                            f"at iteration {i+1} for {image_path.name}. Continuing sampling..."
                        )

            # Track iteration state
            iteration_history.append(
                IterationSnapshot(
                    iteration=len(results),
                    tsv_data=aggregated_tsv if aggregated_tsv else "",
                    converged=has_converged_now,
                )
            )

            # Break if converged
            if converged:
                break

        # Final aggregation
        aggregated_tsv, variance_map, _ = self.aggregation.aggregate(results)

        total_input_tokens = sum(r.input_tokens for r in results)
        total_output_tokens = sum(r.output_tokens for r in results)

        return EnsembleResult(
            tsv_data=aggregated_tsv,
            num_samples=len(results),
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            converged=converged,
            convergence_iteration=convergence_iteration,
            variance_map=variance_map,
            metadata={
                "min_samples": self.min_samples,
                "max_samples": self.max_samples,
            },
            iteration_history=iteration_history,
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "strategy": "adaptive",
            "min_samples": self.min_samples,
            "max_samples": self.max_samples,
            "model": self.model,
            "temperature": self.temperature,
            "convergence_checker": self.convergence_checker.get_config(),
        }


class DiversitySamplingStrategy(SamplingStrategy):
    """Sample across different models and prompts."""

    def __init__(
        self,
        models: List[str],
        prompt_variants: Optional[List[str]] = None,
        samples_per_config: int = 2,
        temperature: Optional[float] = None,
        aggregation: Optional[AggregationMethod] = None,
    ):
        """
        Args:
            models: List of models to use
            prompt_variants: List of prompt variant names
            samples_per_config: Samples per (model, prompt) combination
            temperature: Optional temperature override
            aggregation: Aggregation method
        """
        self.models = models
        self.prompt_variants = prompt_variants or ["standard"]
        self.samples_per_config = samples_per_config
        self.temperature = temperature

        if aggregation is None:
            from .aggregation import MedianAggregation

            aggregation = MedianAggregation()
        self.aggregation = aggregation

    def extract(
        self,
        image_path: Path,
        convergence_checker: Optional[ConvergenceChecker] = None,
    ) -> EnsembleResult:
        """Extract using diverse models and prompts."""
        results: List[ExtractionResult] = []
        iteration_history: List[IterationSnapshot] = []

        # TODO: Implement prompt variants
        # For now, just use standard prompt with multiple models

        for model in self.models:
            for prompt_variant in self.prompt_variants:
                for sample_idx in range(self.samples_per_config):
                    extraction = extract_data_from_chart(
                        image_path,
                        model=model,
                        temperature=self.temperature,
                    )

                    result = ExtractionResult(
                        tsv_data=extraction["csv_data"],
                        input_tokens=extraction["input_tokens"],
                        output_tokens=extraction["output_tokens"],
                        model=model,
                        prompt_variant=prompt_variant,
                        metadata={
                            "sample_idx": sample_idx,
                        },
                    )
                    results.append(result)

                    # Track cumulative ensemble state
                    iter_tsv, _, _ = self.aggregation.aggregate(results)
                    iteration_history.append(
                        IterationSnapshot(
                            iteration=len(results),
                            tsv_data=iter_tsv if iter_tsv else "",
                            converged=False,  # Diversity strategy doesn't check convergence
                        )
                    )

        # Final aggregation
        aggregated_tsv, variance_map, _ = self.aggregation.aggregate(results)

        total_input_tokens = sum(r.input_tokens for r in results)
        total_output_tokens = sum(r.output_tokens for r in results)

        return EnsembleResult(
            tsv_data=aggregated_tsv,
            num_samples=len(results),
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            converged=False,
            variance_map=variance_map,
            metadata={
                "num_models": len(self.models),
                "num_prompt_variants": len(self.prompt_variants),
                "samples_per_config": self.samples_per_config,
            },
            iteration_history=iteration_history,
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "strategy": "diversity",
            "models": self.models,
            "prompt_variants": self.prompt_variants,
            "samples_per_config": self.samples_per_config,
            "temperature": self.temperature,
        }


class IncrementalEnsembleSamplingStrategy(SamplingStrategy):
    """Incremental ensemble: extract N, ensemble, add 1, ensemble, compare, stop if same.

    Algorithm:
    1. Extract initial_batch_size tables (e.g., 3)
    2. Ensemble them → ensemble_prev
    3. Extract 1 more table
    4. Ensemble all tables so far → ensemble_curr
    5. If ensemble_prev == ensemble_curr, converged → stop
    6. Else, set ensemble_prev = ensemble_curr and repeat from step 3
    7. Stop at max_samples
    """

    def __init__(
        self,
        initial_batch_size: int,
        max_samples: int,
        model: str,
        temperature: Optional[float] = None,
        aggregation: Optional[AggregationMethod] = None,
        patience: int = 1,
        exact_match: bool = True,
        convergence_coverage: float = 0.95,
        convergence_tolerance: float = 0.01,
    ):
        """
        Args:
            initial_batch_size: Number of initial samples to extract before first ensemble
            max_samples: Maximum number of samples to extract
            model: Model to use for extraction
            temperature: Optional temperature override
            aggregation: Aggregation method (default: MedianAggregation)
            patience: Number of consecutive stable ensemble comparisons required to converge (default: 1)
                     Example with patience=2, initial_batch_size=2:
                     - Initial: ensemble(samples 0,1)
                     - Add sample 2: compare ensemble(0,1) vs ensemble(0,1,2) → if equal, count=1
                     - Add sample 3: compare ensemble(0,1,2) vs ensemble(0,1,2,3) → if equal, count=2 → CONVERGE (at 4 samples total)
            exact_match: Whether to use exact match for convergence check (default: True)
        """
        self.initial_batch_size = initial_batch_size
        self.max_samples = max_samples
        self.model = model
        self.temperature = temperature
        self.patience = patience
        self.exact_match = exact_match
        self.convergence_coverage = convergence_coverage
        self.convergence_tolerance = convergence_tolerance

        if aggregation is None:
            from .aggregation import MedianAggregation

            aggregation = MedianAggregation()
        self.aggregation = aggregation

    def _check_stability(self, ensemble_prev: str, ensemble_curr: str) -> bool:
        if self.exact_match:
            return ensemble_prev == ensemble_curr
        else:
            # median doesn't change by more than 1% of the median cell value for 95% of cells
            try:
                # Parse TSV tables into row-column-value dicts
                def parse_tsv(tsv):
                    lines = [line for line in tsv.strip().splitlines() if line.strip()]
                    if len(lines) < 2:
                        return {}
                    headers = lines[0].split("\t")
                    content = {}
                    for line in lines[1:]:
                        vals = line.split("\t")
                        row_label = vals[0]
                        for col_label, val in zip(headers[1:], vals[1:]):
                            content[(row_label.strip(), col_label.strip())] = (
                                val.strip()
                            )
                    return content

                prev_cells = parse_tsv(ensemble_prev if ensemble_prev else "")
                curr_cells = parse_tsv(ensemble_curr if ensemble_curr else "")

                if set(prev_cells.keys()) != set(curr_cells.keys()):
                    return False
                else:
                    shared_keys = set(prev_cells.keys())
                    num_cells = len(shared_keys)
                    num_within_threshold = 0
                    for k in shared_keys:
                        prev_val = prev_cells[k]
                        curr_val = curr_cells[k]
                        try:
                            prev_f = float(prev_val)
                            curr_f = float(curr_val)
                        except Exception:
                            continue  # skip non-numeric cells

                        median_val = prev_f  # We compare the current value to the previous median
                        # If the cell values are nearly zero, treat tiny changes as "stable"
                        abs_tol = max(0.1, self.convergence_tolerance * abs(median_val))
                        if abs(prev_f - curr_f) <= abs_tol:
                            num_within_threshold += 1

                    cell_fraction = (
                        (num_within_threshold / num_cells) if num_cells > 0 else 0.0
                    )
                    return cell_fraction >= self.convergence_coverage
            except Exception as e:
                logger.warning(f"Could not compute stability: {e}")
                return False

    def extract(
        self,
        image_path: Path,
        convergence_checker: Optional[ConvergenceChecker] = None,
    ) -> EnsembleResult:
        """Extract with incremental ensemble convergence."""
        results: List[ExtractionResult] = []
        converged = False
        convergence_iteration = None
        ensemble_prev = None
        consecutive_stable = 0  # Track consecutive stable ensembles
        fuzzy_match_failures = (
            0  # Track number of fuzzy matching failures for this image
        )
        iteration_history: List[IterationSnapshot] = (
            []
        )  # Track ensemble state at each iteration

        # Phase 1: Extract initial batch
        logger.debug(
            f"Incremental ensemble: Extracting initial batch of {self.initial_batch_size} samples"
        )
        for i in range(self.initial_batch_size):
            extraction = extract_data_from_chart(
                image_path,
                model=self.model,
                temperature=self.temperature,
            )

            result = ExtractionResult(
                tsv_data=extraction["csv_data"],
                input_tokens=extraction["input_tokens"],
                output_tokens=extraction["output_tokens"],
                model=self.model,
                metadata={"sample_idx": i},
            )
            results.append(result)

            if extraction["csv_data"] is None:
                logger.warning(
                    f"Incremental ensemble: Sample {i+1}/{self.initial_batch_size} failed for {image_path.name}"
                )

        # Ensemble initial batch
        ensemble_prev, variance_map_prev, _, mad_tsv = self.aggregation.aggregate(
            results
        )

        # Save initial snapshot
        iteration_history.append(
            IterationSnapshot(
                iteration=len(results),
                tsv_data=ensemble_prev if ensemble_prev else "",
                converged=False,
            )
        )

        # Check if initial ensemble is valid (non-empty and has data rows, not just header)
        has_initial_content = (
            ensemble_prev
            and ensemble_prev.strip()
            and len(ensemble_prev.strip().split("\n")) > 1
        )

        # Phase 2: Add one sample at a time and check convergence
        # If initial ensemble is empty, keep sampling until we get a non-empty result
        starting_iteration = self.initial_batch_size

        if not has_initial_content:
            logger.warning(
                f"Incremental ensemble: Initial ensemble empty for {image_path.name}, "
                f"continuing to sample until non-empty result found"
            )
        else:
            logger.debug(
                f"Incremental ensemble: Initial ensemble completed with {len(results)} samples"
            )

        for i in range(starting_iteration, self.max_samples):
            # Extract one more sample
            extraction = extract_data_from_chart(
                image_path,
                model=self.model,
                temperature=self.temperature,
            )

            result = ExtractionResult(
                tsv_data=extraction["csv_data"],
                input_tokens=extraction["input_tokens"],
                output_tokens=extraction["output_tokens"],
                model=self.model,
                metadata={"sample_idx": i},
            )
            results.append(result)

            if extraction["csv_data"] is None:
                logger.warning(
                    f"Incremental ensemble: Sample {i+1} failed for {image_path.name}"
                )

            # Ensemble all samples so far, using previous ensemble as fallback
            ensemble_curr, variance_map_curr, used_fallback, mad_tsv = (
                self.aggregation.aggregate(
                    results,
                    fallback_tsv=ensemble_prev,  # Use last successful ensemble if fuzzy matching fails
                )
            )

            # Save snapshot for this iteration
            is_stable = self._check_stability(ensemble_prev, ensemble_curr)
            iteration_history.append(
                IterationSnapshot(
                    iteration=len(results),
                    tsv_data=ensemble_curr if ensemble_curr else "",
                    converged=is_stable and not used_fallback,
                )
            )

            if not ensemble_curr or not ensemble_curr.strip():
                logger.warning(
                    f"Incremental ensemble: Ensemble failed at iteration {i+1} for {image_path.name}, "
                    f"continuing sampling..."
                )
                continue

            # Check if current ensemble has content
            has_content = (
                len(ensemble_curr.strip().split("\n")) > 1
            )  # More than just header row

            # If this is the first non-empty ensemble (initial was empty), use it as baseline
            if not has_initial_content and has_content:
                logger.info(
                    f"Incremental ensemble: Found first non-empty result at iteration {i+1} "
                    f"for {image_path.name}, using as baseline"
                )
                has_initial_content = True  # Mark that we now have a baseline
                ensemble_prev = ensemble_curr
                variance_map_prev = variance_map_curr
                consecutive_stable = 0
                continue  # Skip convergence check, just set baseline

            # Check if ensemble is stable (prev == curr)
            # CRITICAL CONDITIONS for convergence:
            # 1. Fuzzy matching succeeded (not using fallback)
            # 2. Ensemble has actual content (not just header)
            # 3. We have a valid baseline to compare against
            # The used_fallback flag prevents false convergence from fallback loops
            is_stable = self._check_stability(ensemble_prev, ensemble_curr)

            is_genuine_convergence = (
                is_stable and has_content and not used_fallback and has_initial_content
            )

            if is_genuine_convergence:
                # Increment consecutive stable count
                # This counts how many times IN A ROW the ensemble has remained unchanged
                # With patience=N, we need N consecutive stable comparisons to converge
                consecutive_stable += 1
                logger.debug(
                    f"Incremental ensemble: Ensemble stable ({consecutive_stable}/{self.patience}) "
                    f"at iteration {i+1} for {image_path.name}"
                )

                # Check if we've reached patience threshold
                # This means we've seen patience consecutive stable ensemble comparisons
                if consecutive_stable >= self.patience:
                    converged = True
                    convergence_iteration = (
                        i + 1
                    )  # Total number of samples when converged
                    logger.debug(
                        f"Incremental ensemble: Converged at iteration {convergence_iteration} "
                        f"for {image_path.name} (ensemble stable for {self.patience} consecutive comparisons)"
                    )
                    break
            elif used_fallback:
                # Fuzzy matching failed, using fallback - don't count as convergence
                fuzzy_match_failures += 1  # Increment failure counter
                # logger.warning(
                #     f"Incremental ensemble: Fuzzy matching failed at iteration {i+1}, "
                #     f"using fallback for {image_path.name}. Continuing sampling..."
                # )
                consecutive_stable = 0
            elif is_stable and not has_content:
                # Ensemble is stable but empty - this is bad, reset counter and continue
                logger.warning(
                    f"Incremental ensemble: Ensemble stable but empty/invalid at iteration {i+1} "
                    f"for {image_path.name}, continuing sampling..."
                )
                consecutive_stable = 0
            else:
                # Ensemble changed, reset counter
                consecutive_stable = 0

            # Update for next iteration
            ensemble_prev = ensemble_curr
            variance_map_prev = variance_map_curr

        # Final aggregation (use last ensemble result)
        aggregated_tsv = ensemble_prev if ensemble_prev else ""
        variance_map = variance_map_prev if variance_map_prev else {}

        # Log warning if we never found a non-empty result
        if not has_initial_content:
            logger.warning(
                f"Incremental ensemble: Exhausted all {len(results)} samples without finding "
                f"non-empty result for {image_path.name}"
            )

        total_input_tokens = sum(r.input_tokens for r in results)
        total_output_tokens = sum(r.output_tokens for r in results)

        return EnsembleResult(
            tsv_data=aggregated_tsv,
            num_samples=len(results),
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            converged=converged,
            convergence_iteration=convergence_iteration,
            variance_map=variance_map,
            mad_tsv=mad_tsv,
            metadata={
                "initial_batch_size": self.initial_batch_size,
                "max_samples": self.max_samples,
                "patience": self.patience,
                "fuzzy_match_failures": fuzzy_match_failures,
            },
            iteration_history=iteration_history,
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "strategy": "incremental_ensemble",
            "initial_batch_size": self.initial_batch_size,
            "max_samples": self.max_samples,
            "model": self.model,
            "temperature": self.temperature,
            "patience": self.patience,
        }


class StructuredSamplingStrategy(SamplingStrategy):
    """Structured sampling

    Algorithm:
    1. Use expensive model to determine the structure of the TSVs (ie row and column labels)
    1. Extract initial_batch_size tables (e.g., 3) according to the structure
    2. Ensemble them
    3. Extract 1 more table
    4. Ensemble all tables so far
    5. If converged → stop
    6. Else, repeat from step 3
    7. Stop at max_samples
    """

    def __init__(
        self,
        initial_batch_size: int,
        max_samples: int,
        model: str,
        structure_model: str,
        temperature: Optional[float] = None,
        aggregation: Optional[AggregationMethod] = None,
        patience: int = 1,
    ):
        """
        Args:
            initial_batch_size: Number of initial samples to extract before first ensemble
            max_samples: Maximum number of samples to extract
            model: Model to use for extraction
            structure_model: Model to use for determining the structure of the TSVs
            temperature: Optional temperature override
            aggregation: Aggregation method (default: MedianAggregation)
            patience: Number of consecutive stable ensemble comparisons required to converge (default: 1)
                     Example with patience=2, initial_batch_size=2:
                     - Initial: ensemble(samples 0,1)
                     - Add sample 2: compare ensemble(0,1) vs ensemble(0,1,2) → if equal, count=1
                     - Add sample 3: compare ensemble(0,1,2) vs ensemble(0,1,2,3) → if equal, count=2 → CONVERGE (at 4 samples total)
        """
        self.initial_batch_size = initial_batch_size
        self.max_samples = max_samples
        self.model = model
        self.structure_model = structure_model
        self.temperature = temperature
        self.patience = patience

        if aggregation is None:
            from .aggregation import MedianAggregation

            aggregation = MedianAggregation()
        self.aggregation = aggregation

    def _check_stability(self, ensemble_prev: str, ensemble_curr: str) -> bool:
        # median doesn't change by more than 1% of the median cell value for 95% of cells
        try:
            # Parse TSV tables into row-column-value dicts
            def parse_tsv(tsv):
                lines = [line for line in tsv.strip().splitlines() if line.strip()]
                if len(lines) < 2:
                    return {}
                headers = lines[0].split("\t")
                content = {}
                for line in lines[1:]:
                    vals = line.split("\t")
                    row_label = vals[0]
                    for col_label, val in zip(headers[1:], vals[1:]):
                        content[(row_label.strip(), col_label.strip())] = val.strip()
                return content

            prev_cells = parse_tsv(ensemble_prev if ensemble_prev else "")
            curr_cells = parse_tsv(ensemble_curr if ensemble_curr else "")

            if set(prev_cells.keys()) != set(curr_cells.keys()):
                return False
            else:
                shared_keys = set(prev_cells.keys())
                num_cells = len(shared_keys)
                num_within_threshold = 0
                for k in shared_keys:
                    prev_val = prev_cells[k]
                    curr_val = curr_cells[k]
                    try:
                        prev_f = float(prev_val)
                        curr_f = float(curr_val)
                    except Exception:
                        continue  # skip non-numeric cells

                    median_val = (
                        prev_f  # We compare the current value to the previous median
                    )
                    # If the cell values are nearly zero, treat tiny changes as "stable"
                    abs_tol = max(0.1, 0.01 * abs(median_val))  # 1%
                    if abs(prev_f - curr_f) <= abs_tol:
                        num_within_threshold += 1

                cell_fraction = (
                    (num_within_threshold / num_cells) if num_cells > 0 else 0.0
                )
                return cell_fraction >= 0.95
        except Exception as e:
            logger.warning(f"Could not compute stability: {e}")
            return False

    def extract(
        self,
        image_path: Path,
        convergence_checker: Optional[ConvergenceChecker] = None,
    ) -> EnsembleResult:
        """Extract with incremental ensemble convergence."""
        results: List[ExtractionResult] = []
        converged = False
        convergence_iteration = None
        ensemble_prev = None
        consecutive_stable = 0  # Track consecutive stable ensembles

        iteration_history: List[IterationSnapshot] = (
            []
        )  # Track ensemble state at each iteration

        # Extract structure
        structure = extract_structure_from_chart(image_path, model=self.structure_model)
        structure = structure["structure"]

        # Phase 1: Extract initial batch
        logger.debug(
            f"Structured sampling: Extracting initial batch of {self.initial_batch_size} samples"
        )
        for i in range(self.initial_batch_size):
            extraction = extract_data_from_chart(
                image_path,
                model=self.model,
                temperature=self.temperature,
                structure=structure,
            )

            result = ExtractionResult(
                tsv_data=extraction["csv_data"],
                input_tokens=extraction["input_tokens"],
                output_tokens=extraction["output_tokens"],
                model=self.model,
                metadata={"sample_idx": i},
            )
            results.append(result)

            if extraction["csv_data"] is None:
                logger.warning(
                    f"Incremental ensemble: Sample {i+1}/{self.initial_batch_size} failed for {image_path.name}"
                )

        for result in results:
            if not conforms_to_structure(result.tsv_data, structure):
                logger.error(
                    f"Structured sampling: Sample {result.metadata['sample_idx']} does not conform to structure for {image_path.name}"
                )
                raise ValueError(
                    f"Sample {result.metadata['sample_idx']} does not conform to structure for {image_path.name}"
                )

        # Ensemble initial batch
        ensemble_prev, variance_map_prev, _, _ = self.aggregation.aggregate(
            results, structure=structure
        )

        # Save initial snapshot
        iteration_history.append(
            IterationSnapshot(
                iteration=len(results),
                tsv_data=ensemble_prev if ensemble_prev else "",
                converged=False,
            )
        )

        # Check if initial ensemble is valid (non-empty and has data rows, not just header)
        has_initial_content = (
            ensemble_prev
            and ensemble_prev.strip()
            and len(ensemble_prev.strip().split("\n")) > 1
        )

        # Phase 2: Add one sample at a time and check convergence
        # If initial ensemble is empty, keep sampling until we get a non-empty result
        starting_iteration = self.initial_batch_size

        if not has_initial_content:
            logger.warning(
                f"Incremental ensemble: Initial ensemble empty for {image_path.name}, "
                f"continuing to sample until non-empty result found"
            )
        else:
            logger.debug(
                f"Incremental ensemble: Initial ensemble completed with {len(results)} samples"
            )

        for i in range(starting_iteration, self.max_samples):
            # Extract one more sample
            extraction = extract_data_from_chart(
                image_path,
                model=self.model,
                temperature=self.temperature,
                structure=structure,
            )

            result = ExtractionResult(
                tsv_data=extraction["csv_data"],
                input_tokens=extraction["input_tokens"],
                output_tokens=extraction["output_tokens"],
                model=self.model,
                metadata={"sample_idx": i},
            )
            results.append(result)

            if extraction["csv_data"] is None:
                logger.warning(
                    f"Incremental ensemble: Sample {i+1} failed for {image_path.name}"
                )

            # Ensemble all samples so far, using previous ensemble as fallback
            ensemble_curr, variance_map_curr, used_fallback, _ = (
                self.aggregation.aggregate(
                    results,
                    fallback_tsv=ensemble_prev,  # Use last successful ensemble if fuzzy matching fails
                    structure=structure,
                )
            )

            # Save snapshot for this iteration
            is_stable = self._check_stability(ensemble_prev, ensemble_curr)
            iteration_history.append(
                IterationSnapshot(
                    iteration=len(results),
                    tsv_data=ensemble_curr if ensemble_curr else "",
                    converged=is_stable and not used_fallback,
                )
            )

            if not ensemble_curr or not ensemble_curr.strip():
                logger.warning(
                    f"Incremental ensemble: Ensemble failed at iteration {i+1} for {image_path.name}, "
                    f"continuing sampling..."
                )
                continue

            # Check if current ensemble has content
            has_content = (
                len(ensemble_curr.strip().split("\n")) > 1
            )  # More than just header row

            # If this is the first non-empty ensemble (initial was empty), use it as baseline
            if not has_initial_content and has_content:
                logger.info(
                    f"Incremental ensemble: Found first non-empty result at iteration {i+1} "
                    f"for {image_path.name}, using as baseline"
                )
                has_initial_content = True  # Mark that we now have a baseline
                ensemble_prev = ensemble_curr
                variance_map_prev = variance_map_curr
                consecutive_stable = 0
                continue  # Skip convergence check, just set baseline

            # Check if ensemble is stable (prev == curr)
            # CRITICAL CONDITIONS for convergence:
            # 1. Fuzzy matching succeeded (not using fallback)
            # 2. Ensemble has actual content (not just header)
            # 3. We have a valid baseline to compare against
            # The used_fallback flag prevents false convergence from fallback loops
            is_stable = self._check_stability(ensemble_prev, ensemble_curr)

            is_genuine_convergence = (
                is_stable and has_content and not used_fallback and has_initial_content
            )

            if is_genuine_convergence:
                # Increment consecutive stable count
                # This counts how many times IN A ROW the ensemble has remained unchanged
                # With patience=N, we need N consecutive stable comparisons to converge
                consecutive_stable += 1
                logger.debug(
                    f"Incremental ensemble: Ensemble stable ({consecutive_stable}/{self.patience}) "
                    f"at iteration {i+1} for {image_path.name}"
                )

                # Check if we've reached patience threshold
                # This means we've seen patience consecutive stable ensemble comparisons
                if consecutive_stable >= self.patience:
                    converged = True
                    convergence_iteration = (
                        i + 1
                    )  # Total number of samples when converged
                    logger.debug(
                        f"Incremental ensemble: Converged at iteration {convergence_iteration} "
                        f"for {image_path.name} (ensemble stable for {self.patience} consecutive comparisons)"
                    )
                    break
            elif used_fallback:
                consecutive_stable = 0
            elif is_stable and not has_content:
                # Ensemble is stable but empty - this is bad, reset counter and continue
                logger.warning(
                    f"Incremental ensemble: Ensemble stable but empty/invalid at iteration {i+1} "
                    f"for {image_path.name}, continuing sampling..."
                )
                consecutive_stable = 0
            else:
                # Ensemble changed, reset counter
                consecutive_stable = 0

            # Update for next iteration
            ensemble_prev = ensemble_curr
            variance_map_prev = variance_map_curr

        # Final aggregation (use last ensemble result)
        aggregated_tsv = ensemble_prev if ensemble_prev else ""
        variance_map = variance_map_prev if variance_map_prev else {}

        # Log warning if we never found a non-empty result
        if not has_initial_content:
            logger.warning(
                f"Incremental ensemble: Exhausted all {len(results)} samples without finding "
                f"non-empty result for {image_path.name}"
            )

        total_input_tokens = sum(r.input_tokens for r in results)
        total_output_tokens = sum(r.output_tokens for r in results)

        return EnsembleResult(
            tsv_data=aggregated_tsv,
            num_samples=len(results),
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            converged=converged,
            convergence_iteration=convergence_iteration,
            variance_map=variance_map,
            metadata={
                "initial_batch_size": self.initial_batch_size,
                "max_samples": self.max_samples,
                "patience": self.patience,
            },
            iteration_history=iteration_history,
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "strategy": "incremental_ensemble",
            "initial_batch_size": self.initial_batch_size,
            "max_samples": self.max_samples,
            "model": self.model,
            "structure_model": self.structure_model,
            "temperature": self.temperature,
            "patience": self.patience,
        }
