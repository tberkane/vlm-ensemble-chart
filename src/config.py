from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PredictConfig:
    input_images_dir: str
    model: str
    temperature: Optional[float] = None
    prompt: str = "CHART_EXTRACTION_PROMPT"
    run_dir: Optional[str] = None

    # Weights & Biases settings
    use_wandb: bool = False
    wandb_project: str = "vlm-ensemble-chart-extraction"


@dataclass
class EvalConfig:
    run_dir: str
    gt_dir: str

    use_wandb: bool = False
    wandb_project: str = "vlm-ensemble-chart-extraction"


@dataclass
class EnsembleConfig:
    # Directories of member prediction runs
    member_run_dirs: List[str]

    # How to aggregate predictions: "median", "mean", etc.
    aggregation_methods: List[str] = field(default_factory=lambda: ["median"])

    # Threshold for how similar rows have to be to be grouped together (used to be 0.9)
    row_similarity_threshold: float = 0.5
    # Threshold for how similar columns have to be to be grouped together
    column_similarity_threshold: float = 0.5

    # Optional tag for naming the ensemble run dir
    output_tag: Optional[str] = None
