"""
Configuration classes for adaptive ensemble methods.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ConvergenceConfig:
    """Configuration for convergence checking."""

    type: str  # "exact_match" or "variance"

    # ExactMatch params
    num_matches: int = 2

    # Variance params (now uses MAD instead of CV)
    variance_threshold: float = 0.01
    require_all_cells: bool = True
    patience: int = 1  # Number of consecutive iterations MAD must stay below threshold

    # Common params
    min_samples: int = 2


@dataclass
class AdaptivePredictConfig:
    """Configuration for adaptive prediction."""

    # Input/output
    input_images_dir: str
    run_dir: Optional[str] = None
    gt_dir: Optional[str] = (
        None  # Optional ground truth for computing iteration-level metrics
    )

    # Strategy selection
    strategy: str = (
        "adaptive"  # "fixed", "adaptive", "diversity", "incremental_ensemble"
    )

    # Model params
    model: str = "qwen/qwen-2.5-72b-instruct"
    temperature: Optional[float] = None

    # Fixed strategy params
    fixed_num_samples: int = 10

    # Adaptive strategy params
    adaptive_min_samples: int = 2
    adaptive_max_samples: int = 10
    convergence: ConvergenceConfig = field(
        default_factory=lambda: ConvergenceConfig(type="variance")
    )

    # Diversity strategy params
    diversity_models: Optional[List[str]] = None
    diversity_prompt_variants: Optional[List[str]] = None
    diversity_samples_per_config: int = 2

    # Incremental ensemble strategy params
    incremental_initial_batch_size: int = 3  # Number of initial samples to extract
    incremental_max_samples: int = 10  # Maximum number of samples to extract
    incremental_patience: int = (
        1  # Number of consecutive stable ensembles to trigger convergence
    )
    incremental_min_samples_for_convergence: int = (
        5  # Minimum samples before checking convergence (fuzzy matching unstable with few samples)
    )
    incremental_exact_match: bool = True

    # 95% of values should not change by more than 1%
    incremental_convergence_coverage: float = 0.95
    incremental_convergence_tolerance: float = 0.01

    # Aggregation
    aggregation: str = "median"  # "median" or "mean"

    # Fuzzy matching parameters for MedianAggregation
    # (handles row/column name mismatches across extractions)
    row_similarity_threshold: float = 0.5  # Threshold for row label matching (0-1)
    column_similarity_threshold: float = (
        0.5  # Threshold for column label matching (0-1)
    )
    pruning: bool = True
    pruning_threshold: float = 0.5
    structure_model: str = "gpt-5.2"

    # W&B
    use_wandb: bool = False
    wandb_project: str = "vlm-ensemble-chart-extraction"
    wandb_tags: Optional[List[str]] = None


@dataclass
class ComparisonConfig:
    """Configuration for comparing strategies."""

    # Input
    input_images_dir: str
    gt_dir: str
    output_dir: str

    # Strategies to compare
    strategies: List[Dict[str, Any]] = field(default_factory=list)

    # Evaluation params
    max_images: Optional[int] = None  # Limit for testing

    # W&B
    use_wandb: bool = False
    wandb_project: str = "vlm-ensemble-chart-extraction"
