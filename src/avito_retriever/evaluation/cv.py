from __future__ import annotations

import hashlib
import random

import pandas as pd

from avito_retriever.preprocessing.normalize import normalize_whitespace


def make_grouped_query_folds(calibration: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> pd.Series:
    """Assign exact normalized query duplicates to the same deterministic fold."""

    groups: dict[str, list[int]] = {}
    for row_index, text in calibration["query_text"].items():
        normalized = normalize_whitespace(str(text)).casefold().replace("ё", "е")
        group = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
        groups.setdefault(group, []).append(int(row_index))

    rng = random.Random(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)

    fold_sizes = [0] * n_folds
    assignments: dict[int, int] = {}
    for _, indices in group_items:
        fold = min(range(n_folds), key=lambda value: fold_sizes[value])
        for index in indices:
            assignments[index] = fold
        fold_sizes[fold] += len(indices)
    return pd.Series(assignments, name="fold").reindex(calibration.index).astype(int)

