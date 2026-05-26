"""Sample datasets from pruned World Bank data."""

import random
from .types import Dataset


def sample_datasets(
    pruned_data: dict,
    n: int = 1000,
    seed: int = 42,
) -> list[Dataset]:
    """Greedily sample n datasets, each with 1 indicator × 2-3 countries.

    Args:
        pruned_data: {series_name: {country_name: {year: value}}}
        n: number of datasets to sample
        seed: random seed for reproducibility

    Returns:
        List of Dataset objects.
    """
    rng = random.Random(seed)

    # Flat list of all (series, country) pairs
    all_pairs = [
        (series, country)
        for series in pruned_data
        for country in pruned_data[series]
    ]
    rng.shuffle(all_pairs)

    used = set()
    datasets = []
    pair_idx = 0

    while len(datasets) < n and pair_idx < len(all_pairs):
        series, country = all_pairs[pair_idx]
        pair_idx += 1

        if (series, country) in used:
            continue

        country_dict = pruned_data[series]
        available = [c for c in country_dict if (series, c) not in used]
        if not available:
            continue

        k = min(len(available), rng.randint(2, 3))
        chosen = rng.sample(available, k)

        for c in chosen:
            used.add((series, c))

        datasets.append(
            Dataset(
                index=len(datasets) + 1,
                series_name=series,
                countries={c: country_dict[c] for c in chosen},
            )
        )

    print(f"Sampled {len(datasets)} datasets")
    return datasets
