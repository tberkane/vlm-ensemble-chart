# Self-Ensembling Vision-Language Models for Chart Data Extraction

Code and benchmark for the paper *Self-Ensembling Vision-Language Models for Chart Data Extraction*.

We repeatedly sample tabular outputs from a base VLM for a fixed chart image, align the
candidate tables at the cell level, and take per-cell medians to produce a more accurate
consensus table. The method also includes convergence-based early stopping and an
uncertainty estimate from dispersion across samples. It is model-agnostic and can be
layered on top of any chart-to-table model.

This repository also contains **WB-ChartExtract**, a benchmark of 1,000 synthetic charts
built from World Bank data (4 chart types × 4 rendering libraries), with clean
ground-truth tables.

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

Download WB-ChartExtract from the Hugging Face Hub and place it under `data/`:

```bash
hf download tberkane/WB-ChartExtract --repo-type dataset --local-dir "data/WB-ChartExtract"
```

This yields:

```
data/WB-ChartExtract/
├── png/            # 1,000 chart images (1.png ... 1000.png)
├── tables/         # 1,000 ground-truth tables (1.csv ... 1000.csv)
└── metadata.json   # chart_type, library, countries, series, etc. per image
```

For ChartQA, download the dataset from
[ahmed-masry/ChartQA](https://huggingface.co/datasets/ahmed-masry/ChartQA) and arrange the
test split as `data/ChartQA/png` and `data/ChartQA/tables`.

## Usage

All scripts take a YAML config via `--config`. Edit the configs in `configs/` to change the
base model, temperature, data paths, or ensembling hyperparameters.

**Single-pass extraction:**

```bash
python scripts/predict.py --config configs/predict/wb_scout.yaml
```

**Self-ensembling (iterative sampling + cell-wise aggregation + convergence detection):**

```bash
python scripts/adaptive/predict.py --config configs/adaptive/wb_scout.yaml
```

**Evaluation (RMS_F1):** point `run_dir` at a prediction/ensemble output directory, then:

```bash
python scripts/eval.py --config configs/eval/wb.yaml
```

**Regenerate the WB-ChartExtract benchmark from scratch:**

```bash
python scripts/generate_world_bank_charts.py
```

Configs are provided for both ChartQA and WB-ChartExtract. To use a different base model,
change the `model` field (e.g. `qwen/qwen3-vl-235b-a22b-instruct`, `tinychart`, `deplot`).
On WB-ChartExtract the paper uses temperature 0.0; on ChartQA, 2.0 for self-ensembling.

## Repository structure

```
src/
├── extract_data.py        # query a base VLM for a TSV table (with response caching)
├── ensemble.py            # cell-level table alignment + aggregation
├── eval_chart2table.py    # RMS_F1 evaluation metric
├── adaptive/              # iterative sampling strategies, convergence, aggregators
└── chart_gen/             # WB-ChartExtract chart generation (4 libraries, 4 chart types)
scripts/                   # predict / ensemble / eval / benchmark-generation entry points
experiments/               # scripts reproducing the paper's figures and tables
configs/                   # example configs for prediction, self-ensembling, and eval
```

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
