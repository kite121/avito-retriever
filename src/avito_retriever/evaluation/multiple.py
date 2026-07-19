from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    """Return Holm-adjusted p-values in the input order."""

    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted_sorted = np.empty(len(values), dtype=float)
    running = 0.0
    total = len(values)
    for position, original_index in enumerate(order):
        adjusted = (total - position) * values[original_index]
        running = max(running, adjusted)
        adjusted_sorted[position] = min(1.0, running)
    result = np.empty(len(values), dtype=float)
    result[order] = adjusted_sorted
    return result.tolist()

