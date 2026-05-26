# Self-Ensembling Vision-Language Models for Chart Data Extraction

Core implementation for the paper *Self-Ensembling Vision-Language Models for Chart Data
Extraction*.

We repeatedly sample tabular outputs from a base VLM for a fixed chart image, align the
candidate tables at the cell level, and take per-cell medians to produce a more accurate
consensus table. The method also includes convergence-based early stopping and an
uncertainty estimate from dispersion across samples. It is model-agnostic and can be
layered on top of any chart-to-table model.

This work also introduces **WB-ChartExtract**, a benchmark of 1,000 synthetic charts built
from World Bank data (4 chart types × 4 rendering libraries) with clean ground-truth tables.

- **Code:** https://github.com/tberkane/vlm-ensemble-chart
- **Dataset:** https://huggingface.co/datasets/tberkane/WB-ChartExtract

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Copy `.env.example` to `.env` and add your API keys for the providers you intend to use
(Groq, OpenRouter, OpenAI, Anthropic, Gemini):

```bash
cp .env.example .env
```

## Data

Download WB-ChartExtract from the Hugging Face Hub:

```bash
hf download tberkane/WB-ChartExtract --repo-type dataset --local-dir "data/WB-ChartExtract"
```

This yields `png/` (1,000 chart images), `tables/` (1,000 ground-truth CSV tables), and
`metadata.json` (chart type, library, countries, series, etc. per image).

## Library layout

```
src/
├── extract_data.py        # query a base VLM for a TSV table (with response caching)
├── ensemble.py            # cell-level table alignment + aggregation
├── eval_chart2table.py    # RMS_F1 evaluation metric (and error-type breakdown)
├── config.py, utils.py    # config dataclasses and table-normalization helpers
├── adaptive/              # iterative self-ensembling
│   ├── strategies.py      #   sampling strategies (incl. IncrementalEnsembleSamplingStrategy)
│   ├── aggregation.py     #   cell-wise aggregators (median, mean, medoid, Huber)
│   └── convergence.py     #   convergence detection / early stopping
└── chart_gen/             # WB-ChartExtract chart generation (4 libraries, 4 chart types)
```

## Programmatic use

Sample a single table from a base VLM:

```python
from pathlib import Path
from src.extract_data import extract_data_from_chart, CHART_EXTRACTION_PROMPT_WB

result = extract_data_from_chart(
    image_path=Path("data/WB-ChartExtract/png/1.png"),
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.0,
    prompt=CHART_EXTRACTION_PROMPT_WB,
)
```

Self-ensembling composes the building blocks in `src.adaptive` — an
`IncrementalEnsembleSamplingStrategy` that repeatedly samples and updates a consensus,
`MedianAggregation` for cell-wise aggregation, and `IncrementalEnsembleConvergence` for
early stopping:

```python
from src.adaptive import (
    IncrementalEnsembleSamplingStrategy,
    MedianAggregation,
    IncrementalEnsembleConvergence,
)
```

On WB-ChartExtract the paper uses temperature 0.0; on ChartQA, 2.0 for self-ensembling. To
use a different base model, change the `model` argument (e.g.
`qwen/qwen3-vl-235b-a22b-instruct`, `tinychart`, `deplot`). Extraction quality is scored
with the RMS_F1 metric in `src/eval_chart2table.py`.

## Citation

```bibtex
@inproceedings{berkane2026selfensembling,
  title     = {Self-Ensembling Vision-Language Models for Chart Data Extraction},
  author    = {Berkane, Thomas and Wang, Qianyi and Majumder, Maimuna S.},
  year      = {2026}
}
```

## License

Code is released under the Apache 2.0 License (see `LICENSE`). The WB-ChartExtract benchmark
is released under CC BY 4.0; it is derived from World Bank Open Data (CC BY 4.0).
