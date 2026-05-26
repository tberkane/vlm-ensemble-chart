"""
Convergence checking methods for adaptive sampling.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from .base import ConvergenceChecker, ExtractionResult

logger = logging.getLogger(__name__)


class ExactMatchConvergence(ConvergenceChecker):
    """Check if last N results are identical.
    
    Only considers non-empty, non-None results as valid for convergence.
    """
    
    def __init__(self, num_matches: int = 2):
        """
        Args:
            num_matches: Number of consecutive identical results needed
        """
        self.num_matches = num_matches
    
    def has_converged(
        self,
        results: List[ExtractionResult],
        variance_map: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> bool:
        """Check if last num_matches results are identical and non-empty.
        
        Returns False if any of the last num_matches results are None or empty,
        preventing premature convergence on failed extractions.
        """
        if len(results) < self.num_matches:
            return False
        
        last_results = results[-self.num_matches:]
        
        # Check that all results are non-None and non-empty
        for r in last_results:
            if r.tsv_data is None or r.tsv_data.strip() == "":
                logger.debug("Convergence check failed: found None or empty result")
                return False
        
        first_tsv = last_results[0].tsv_data
        is_converged = all(r.tsv_data == first_tsv for r in last_results)
        
        if is_converged:
            logger.debug(f"Exact match convergence achieved with {self.num_matches} identical results")
        
        return is_converged
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "exact_match",
            "num_matches": self.num_matches,
        }


class VarianceConvergence(ConvergenceChecker):
    """Check if MAD (Median Absolute Deviation) is below threshold with patience.
    
    Uses MAD instead of CV for robustness to outliers.
    Requires MAD to be below threshold for 'patience' consecutive iterations.
    """
    
    def __init__(
        self,
        threshold: float = 0.01,
        require_all_cells: bool = True,
        min_samples: int = 2,
        patience: int = 1,
    ):
        """
        Args:
            threshold: Maximum MAD allowed per cell
            require_all_cells: If True, all cells must be below threshold
            min_samples: Minimum samples before checking convergence
            patience: Number of consecutive iterations MAD must stay below threshold
        """
        self.threshold = threshold
        self.require_all_cells = require_all_cells
        self.min_samples = min_samples
        self.patience = patience
        self._consecutive_successes = 0  # Track consecutive low-MAD iterations
    
    def has_converged(
        self,
        results: List[ExtractionResult],
        variance_map: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> bool:
        """Check if MAD is below threshold for patience consecutive iterations.
        
        Returns False if variance_map is None or empty (indicating failed aggregation).
        """
        if len(results) < self.min_samples:
            return False
        
        # Check if we have valid variance data
        if variance_map is None:
            logger.debug("Convergence check failed: variance_map is None")
            self._consecutive_successes = 0
            return False
        
        if not variance_map:
            # Empty variance_map indicates all extractions failed or no valid data
            logger.debug("Convergence check failed: variance_map is empty (no valid data)")
            self._consecutive_successes = 0
            return False
        
        mads = list(variance_map.values())
        
        # Check if current iteration meets threshold
        if self.require_all_cells:
            threshold_met = all(mad <= self.threshold for mad in mads)
        else:
            # Check if mean MAD is below threshold
            mean_mad = sum(mads) / len(mads)
            threshold_met = mean_mad <= self.threshold
        
        # Update patience counter
        if threshold_met:
            self._consecutive_successes += 1
            logger.debug(
                f"MAD threshold met ({self._consecutive_successes}/{self.patience} consecutive)"
            )
        else:
            logger.debug("MAD threshold not met, resetting patience counter")
            self._consecutive_successes = 0  # Reset on failure
        
        # Converge only if threshold met for 'patience' consecutive iterations
        return self._consecutive_successes >= self.patience
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "variance",
            "threshold": self.threshold,
            "require_all_cells": self.require_all_cells,
            "min_samples": self.min_samples,
            "patience": self.patience,
        }


class IncrementalEnsembleConvergence(ConvergenceChecker):
    """Check if ensemble of N samples equals ensemble of N+1 samples.
    
    This is used by IncrementalEnsembleSamplingStrategy to determine when to stop.
    Not meant to be used directly with other sampling strategies.
    """
    
    def __init__(self):
        """Initialize the convergence checker."""
        pass
    
    def has_converged(
        self,
        results: List[ExtractionResult],
        variance_map: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> bool:
        """This is not used by IncrementalEnsembleSamplingStrategy.
        
        The strategy handles convergence checking internally by comparing
        ensemble results directly. This method always returns False.
        """
        return False
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "incremental_ensemble",
        }
