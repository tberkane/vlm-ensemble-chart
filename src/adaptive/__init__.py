"""
Adaptive ensemble extraction methods.

This package provides adaptive, efficient alternatives to fixed-sample ensemble methods.
"""

from .aggregation import (
    HuberAggregation,
    MeanAggregation,
    MedianAggregation,
    MedoidAggregation,
)
from .base import (
    AggregationMethod,
    ConvergenceChecker,
    IterationSnapshot,
    SamplingStrategy,
)
from .convergence import (
    ExactMatchConvergence,
    IncrementalEnsembleConvergence,
    VarianceConvergence,
)
from .strategies import (
    AdaptiveSamplingStrategy,
    DiversitySamplingStrategy,
    FixedSamplingStrategy,
    IncrementalEnsembleSamplingStrategy,
    StructuredSamplingStrategy,
)

__all__ = [
    # Base classes
    "SamplingStrategy",
    "AggregationMethod",
    "ConvergenceChecker",
    "IterationSnapshot",
    # Strategies
    "FixedSamplingStrategy",
    "AdaptiveSamplingStrategy",
    "DiversitySamplingStrategy",
    "IncrementalEnsembleSamplingStrategy",
    "StructuredSamplingStrategy",
    # Aggregation
    "MedianAggregation",
    "MeanAggregation",
    "MedoidAggregation",
    "HuberAggregation",
    # Convergence
    "VarianceConvergence",
    "ExactMatchConvergence",
    "IncrementalEnsembleConvergence",
]
