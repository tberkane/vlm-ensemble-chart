#!/usr/bin/env python3
"""
Generate WB-ChartExtract v2: 1000 charts across 4 types × 4 libraries.

Usage:
    python scripts/generate_world_bank_charts.py \
        --output_dir data/World_Bank_v2/test \
        --n_charts 1000 \
        --seed 42

    # Smoke test (one chart per combination):
    python scripts/generate_world_bank_charts.py \
        --output_dir /tmp/wb_smoke --n_charts 16 --seed 42
"""

import argparse
import csv
import itertools
import json
import random
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.chart_gen.types import CHART_TYPES, LIBRARIES, ChartSpec, ChartMetadata
from src.chart_gen.data_loader import load_and_prune
from src.chart_gen.sampler import sample_datasets
from src.chart_gen.bar_utils import subsample_years
from src.chart_gen.style import (
    random_matplotlib_style,
    random_seaborn_style,
    random_plotly_style,
    random_bokeh_style,
)
from src.chart_gen.renderers import get_renderer


STYLE_FN = {
    "matplotlib": random_matplotlib_style,
    "seaborn": random_seaborn_style,
    "plotly": random_plotly_style,
    "bokeh": random_bokeh_style,
}


def assign_combinations(n: int, seed: int) -> list[tuple[str, str]]:
    """Return n (chart_type, library) pairs, uniformly distributed."""
    rng = random.Random(seed + 1)  # offset so assignment differs from sampling
    combos = list(itertools.product(CHART_TYPES, LIBRARIES))
    # Tile to cover n
    assignments = (combos * ((n // len(combos)) + 1))[:n]
    rng.shuffle(assignments)
    return assignments


def get_years(dataset, chart_type: str) -> tuple[list[int], bool]:
    """Get years to plot. Subsample for bar charts."""
    years_set = set()
    for country_data in dataset.countries.values():
        years_set.update(country_data.keys())
    all_years = sorted(years_set)

    # Remove years where ALL countries are None
    filtered = []
    for yr in all_years:
        if any(dataset.countries[c].get(yr) is not None for c in dataset.countries):
            filtered.append(yr)

    if chart_type in ("grouped_bar", "stacked_bar"):
        subsampled = subsample_years(filtered, max_bars=15)
        return subsampled, len(subsampled) < len(filtered)
    return filtered, False


def write_gt_csv(spec: ChartSpec, csv_path: Path) -> None:
    """Write ground-truth CSV aligned with spec.years."""
    country_names = list(spec.dataset.countries.keys())
    header = ["Year"] + country_names

    rows = []
    for yr in spec.years:
        row = [yr]
        all_nan = True
        for country in country_names:
            val = spec.dataset.countries[country].get(yr)
            if val is None or str(val).strip() == "":
                row.append("nan")
            else:
                row.append(str(float(val)))
                all_nan = False
        if not all_nan:
            rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate WB-ChartExtract v2")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--n_charts", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--raw_csv",
        type=Path,
        default=project_root / "data" / "World Bank" / "world_bank_raw_data.csv",
    )
    args = parser.parse_args()

    png_dir = args.output_dir / "png"
    tables_dir = args.output_dir / "tables"
    png_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load & prune
    pruned = load_and_prune(args.raw_csv)

    # 2. Sample datasets
    datasets = sample_datasets(pruned, n=args.n_charts, seed=args.seed)

    # 3. Assign chart type × library
    assignments = assign_combinations(len(datasets), args.seed)

    # 4. Render
    rng = random.Random(args.seed + 2)
    metadata = {}
    errors = []

    for dataset, (chart_type, library) in zip(datasets, assignments):
        idx = dataset.index
        years, subsampled = get_years(dataset, chart_type)

        style = STYLE_FN[library](rng)

        spec = ChartSpec(
            dataset=dataset,
            chart_type=chart_type,
            library=library,
            years=years,
            style=style,
        )

        # Write GT CSV
        csv_path = tables_dir / f"{idx}.csv"
        write_gt_csv(spec, csv_path)

        # Render PNG
        png_path = png_dir / f"{idx}.png"
        try:
            renderer = get_renderer(library)
            renderer.render(spec, str(png_path))
        except Exception as e:
            errors.append((idx, library, chart_type, str(e)))
            print(f"  ERROR chart {idx} ({library}/{chart_type}): {e}")
            continue

        meta = ChartMetadata(
            index=idx,
            chart_type=chart_type,
            library=library,
            series_name=dataset.series_name,
            countries=list(dataset.countries.keys()),
            num_years=len(years),
            subsampled=subsampled,
        )
        metadata[f"{idx}.png"] = {
            "chart_type": meta.chart_type,
            "library": meta.library,
            "series_name": meta.series_name,
            "countries": meta.countries,
            "num_years": meta.num_years,
            "subsampled": meta.subsampled,
        }

        if idx % 50 == 0 or idx <= 5:
            print(f"  [{idx}/{len(datasets)}] {library}/{chart_type} → {png_path.name}")

    # Save metadata
    meta_path = args.output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    # Summary
    print(f"\nGenerated {len(metadata)} charts in {args.output_dir}")
    if errors:
        print(f"  {len(errors)} errors:")
        for idx, lib, ct, msg in errors[:10]:
            print(f"    chart {idx} ({lib}/{ct}): {msg}")

    # Distribution
    from collections import Counter
    type_counts = Counter(m["chart_type"] for m in metadata.values())
    lib_counts = Counter(m["library"] for m in metadata.values())
    print(f"\nBy type:    {dict(type_counts)}")
    print(f"By library: {dict(lib_counts)}")


if __name__ == "__main__":
    main()
