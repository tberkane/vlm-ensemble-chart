"""
Base classes for adaptive ensemble extraction.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ExtractionResult:
    """Single extraction result."""

    tsv_data: str
    input_tokens: int
    output_tokens: int
    model: str
    prompt_variant: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class IterationSnapshot:
    """Snapshot of ensemble state at a specific iteration."""

    iteration: int  # Number of samples used
    tsv_data: str  # Ensemble result at this iteration
    converged: bool  # Whether ensemble was stable at this iteration


@dataclass
class EnsembleResult:
    """Aggregated ensemble result."""

    tsv_data: str
    num_samples: int
    total_input_tokens: int
    total_output_tokens: int
    converged: bool
    convergence_iteration: Optional[int] = None
    variance_map: Optional[Dict[Tuple[int, int], float]] = None
    mad_tsv: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    iteration_history: Optional[List["IterationSnapshot"]] = (
        None  # Track intermediate states
    )


class SamplingStrategy(ABC):
    """Base class for sampling strategies."""

    @abstractmethod
    def extract(
        self,
        image_path: Path,
        convergence_checker: Optional["ConvergenceChecker"] = None,
    ) -> EnsembleResult:
        """
        Extract data from chart using this sampling strategy.

        Args:
            image_path: Path to chart image
            convergence_checker: Optional checker for early stopping

        Returns:
            EnsembleResult with aggregated data and metadata
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return configuration for this strategy."""
        pass


class AggregationMethod(ABC):
    """Base class for aggregation methods."""

    @abstractmethod
    def aggregate(
        self,
        results: List[ExtractionResult],
    ) -> Tuple[str, Optional[Dict[Tuple[int, int], float]]]:
        """
        Aggregate multiple extraction results.

        Args:
            results: List of extraction results to aggregate

        Returns:
            Tuple of (aggregated_tsv, variance_map)
            variance_map is optional and maps (row, col) to variance
        """
        pass


class ConvergenceChecker(ABC):
    """Base class for convergence checking."""

    @abstractmethod
    def has_converged(
        self,
        results: List[ExtractionResult],
        variance_map: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> bool:
        """
        Check if sampling has converged.

        Args:
            results: List of extraction results so far
            variance_map: Optional variance map from aggregation

        Returns:
            True if converged, False otherwise
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return configuration for this checker."""
        pass
