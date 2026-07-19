from __future__ import annotations

import pandas as pd

from avito_retriever.evaluation.cv import make_grouped_query_folds


def make_tune_confirm_split(
    calibration: pd.DataFrame,
    n_folds: int = 5,
    confirm_fold: int = 0,
    seed: int = 42,
) -> pd.DataFrame:
    folds = make_grouped_query_folds(calibration, n_folds=n_folds, seed=seed)
    result = calibration[["query_id"]].copy()
    result["fold"] = folds.to_numpy()
    result["split"] = result["fold"].eq(confirm_fold).map({True: "confirm", False: "tune"})
    return result


def metric_on_split(per_query: pd.DataFrame, split: pd.DataFrame, metric: str, name: str) -> float:
    query_ids = split.loc[split["split"] == name, "query_id"]
    selected = per_query[per_query["query_id"].isin(query_ids)]
    if selected.empty:
        raise ValueError(f"No per-query rows for split={name!r}")
    return float(selected[metric].mean())

