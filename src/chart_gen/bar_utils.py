"""Utilities for bar-chart-specific data preparation."""

import numpy as np


def subsample_years(years: list[int], max_bars: int = 15) -> list[int]:
    """Subsample to at most *max_bars* evenly-spaced years, always keeping first and last."""
    if len(years) <= max_bars:
        return years
    indices = np.linspace(0, len(years) - 1, max_bars, dtype=int)
    return [years[i] for i in indices]
