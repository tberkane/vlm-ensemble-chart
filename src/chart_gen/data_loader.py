"""Load and prune World Bank time-series data."""

import csv
from pathlib import Path


def _has_none_in_middle(series_dict: dict) -> bool:
    """True if there is a None between two valid values."""
    years_sorted = sorted(series_dict.keys())
    values = [series_dict[y] for y in years_sorted]
    start, end = 0, len(values) - 1
    while start < len(values) and values[start] is None:
        start += 1
    while end >= 0 and values[end] is None:
        end -= 1
    if start > end:
        return False
    return any(v is None for v in values[start : end + 1])


def _has_large_number(series_dict: dict, threshold: float = 1_000_000) -> bool:
    return any(v is not None and abs(v) >= threshold for v in series_dict.values())


def load_and_prune(
    csv_path: str | Path,
    prune_threshold: int = 32,
    max_value: float = 1_000_000,
) -> dict:
    """Load World Bank CSV and prune unsuitable series.

    Returns:
        {series_name: {country_name: {year(int): value(float|None)}}}
    """
    csv_path = Path(csv_path)
    series_none_counts = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        years = [int(col.split()[0]) for col in header[4:]]

        for row in reader:
            series_name = row[0]
            country_name = row[2]
            values = row[4:]

            none_count = 0
            country_dict = {}
            for year, value in zip(years, values):
                if value == "" or value is None or value == "..":
                    country_dict[year] = None
                    none_count += 1
                else:
                    country_dict[year] = float(value)

            series_none_counts[(series_name, country_name)] = (none_count, country_dict)

    # Prune
    result = {}
    for (series_name, country_name), (none_count, country_dict) in series_none_counts.items():
        if (
            none_count < prune_threshold
            and not _has_none_in_middle(country_dict)
            and not _has_large_number(country_dict, max_value)
        ):
            result.setdefault(series_name, {})[country_name] = country_dict

    n_series = sum(len(v) for v in result.values())
    print(f"Loaded {len(series_none_counts)} series, {n_series} remain after pruning")
    return result
